from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dual_payload.config import DEFAULT_CONFIG, load_config, validate_config
from dual_payload.models import WatermarkEncoder, WatermarkDecoder
from dual_payload.system import DualPayloadSystem
from dual_payload.stages import apply_training_stage, build_optimizer, NETWORKS
from dual_payload.checkpoints import save_checkpoint, load_checkpoint, load_weights, model_from_file
from dual_payload.losses import CleanLoss
from dual_payload.protocol import ProtocolV1, fp32_inference
from dual_payload.package import Header, encode_package, verify_package
from dual_payload.crypto import encrypt_patient, decrypt_patient, payload_to_bits
from dual_payload.keyed_permutation import CHANNELS, COLOR, PATIENT, inverse_branch, permutation_key
from dual_payload.training import train_step, train_main, evaluate_main


def config_for(stage):
    c=deepcopy(DEFAULT_CONFIG)
    c['train']['stage']=stage
    c['train']['key_transform']=stage=='protocol_eval'
    if stage=='protocol_eval':
        # Synthetic test values only, not proposed experiment parameters.
        c['protocol'].update(beta_c=100.,beta_m=100.,min_moved_c=2,min_moved_m=2)
    return c


@pytest.mark.parametrize('bits',[None,64,128,255,257,256.,True,'256'])
def test_config_rejects_missing_or_non256(tmp_path,bits):
    c=config_for('joint_256')
    if bits is None: del c['model']['message_bits']
    else: c['model']['message_bits']=bits
    path=tmp_path/'config.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='message_bits'): load_config(path)
    with pytest.raises(ValueError,match='256'): WatermarkDecoder(message_bits=bits)


def test_no_missing_or_legacy_stage_and_no_implicit_protocol_parameters(tmp_path):
    for stage in (None,'joint','stage_a_payload_256','legacy'):
        c=config_for('joint_256'); c['train']['stage']=stage
        if stage is None: del c['train']['stage']
        path=tmp_path/'config.json';path.write_text(json.dumps(c))
        with pytest.raises(ValueError,match='stage'): load_config(path)
    with pytest.raises(ValueError): load_config('configs/medical_protocol_eval.json')
    assert load_config('configs/medical_joint_256.json')['model']['message_bits']==256
    assert {p.name for p in Path('configs').glob('*.json')} == {
        'medical_joint_256.json', 'medical_protocol_eval.json'}


def test_payload_stage_rejected_by_config_cli_and_optimizer(tmp_path):
    c=config_for('payload_256')
    path=tmp_path/'retired.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='stage'): validate_config(c)
    with pytest.raises(ValueError,match='stage'): load_config(path)
    with pytest.raises(ValueError,match='stage'):
        train_main(['--config',str(path),'--smoke','--output-dir',str(tmp_path/'forbidden')])
    with pytest.raises(ValueError,match='stage'): build_optimizer(DualPayloadSystem(),c)
    assert not (tmp_path/'forbidden').exists()


@pytest.mark.parametrize('key',['rgb','message','carrier','range'])
@pytest.mark.parametrize('value',[0.,-1.,float('nan'),float('inf')])
def test_joint_rejects_invalid_core_loss(tmp_path,key,value):
    c=config_for('joint_256');c['loss'][key]=value
    with pytest.raises(ValueError,match=f'loss.{key}'): validate_config(c)
    path=tmp_path/'invalid.json';path.write_text(json.dumps(c))
    with pytest.raises(ValueError,match=f'loss.{key}'): load_config(path)


@pytest.mark.parametrize('key',['rgb','message','carrier','range'])
def test_joint_accepts_positive_core_loss_and_zero_auxiliary(tmp_path,key):
    c=config_for('joint_256');c['loss'][key]=.25  # Validator test value only.
    assert c['loss']['chroma']==c['loss']['luma']==0
    validate_config(c)
    path=tmp_path/'valid.json';path.write_text(json.dumps(c))
    assert load_config(path)['loss']==c['loss']


def test_joint_rejects_init_from_before_creating_model(tmp_path,capsys):
    with patch('dual_payload.training.DualPayloadSystem',side_effect=AssertionError('must reject first')):
        with pytest.raises(SystemExit) as error:
            train_main(['--config','configs/medical_joint_256.json',
                        '--init-from',str(tmp_path/'existing.pt')])
    assert error.value.code==2
    assert 'unrecognized arguments: --init-from' in capsys.readouterr().err


def test_joint_optimizer_and_one_backward_updates_without_protocol():
    c=config_for('joint_256');model=DualPayloadSystem(c['model'],c['channel'])
    optimizer=build_optimizer(model,c)
    assert type(optimizer) is torch.optim.Adam
    assert not optimizer.state
    assert c['train']['key_transform'] is False
    expected=[]
    for name in NETWORKS:
        module=getattr(model,name)
        assert module.training
        assert all(p.requires_grad for p in module.parameters())
        expected.extend(module.parameters())
    actual=[p for group in optimizer.param_groups for p in group['params']]
    assert {id(p) for p in actual}=={id(p) for p in expected}
    heads=[model.color_encoder.head,model.watermark_encoder.head,model.color_decoder.chroma_head,model.watermark_decoder.head]
    before=[m.weight.detach().clone() for m in heads]
    calls=[]
    original=torch.autograd.backward
    def count(*a,**kw): calls.append(1);return original(*a,**kw)
    with ExitStack() as stack:
        # Guard both implementation functions and the aliases used by ProtocolV1.
        for target in ('dual_payload.protocol.permute_coefficients',
                       'dual_payload.keyed_permutation.permute_coefficients',
                       'dual_payload.protocol.encrypt_patient','dual_payload.crypto.encrypt_patient',
                       'dual_payload.protocol.encode_package','dual_payload.package.encode_package'):
            stack.enter_context(patch(target,side_effect=AssertionError('protocol entered training')))
        stack.enter_context(patch('torch.autograd.backward',side_effect=count))
        loss_call=stack.enter_context(patch.object(CleanLoss,'forward',autospec=True,side_effect=CleanLoss.forward))
        forward=stack.enter_context(patch.object(model,'forward',wraps=model.forward))
        step=stack.enter_context(patch.object(optimizer,'step',wraps=optimizer.step))
        losses=train_step(model,{'rgb':torch.rand(1,3,256,256),'message':torch.randint(0,2,(1,256)).float()},
                          c,optimizer,CleanLoss(c['loss']),torch.device('cpu'))
    assert len(calls)==forward.call_count==loss_call.call_count==step.call_count==1 and losses['total']>0
    for i,head in enumerate(heads):
        assert not torch.equal(before[i],head.weight)
    # Resume train mode after ordinary validation, with all four networks enabled.
    model.eval();apply_training_stage(model,c)
    assert all(getattr(model,name).training for name in NETWORKS)


@pytest.fixture(scope='module')
def model_file(tmp_path_factory):
    path=tmp_path_factory.mktemp('models')/'v3.pt'
    c=config_for('joint_256')
    model=DualPayloadSystem(c['model'],c['channel'])
    save_checkpoint(path,model,c)
    return path


def test_strict_checkpoint_and_no_legacy_loader(model_file,tmp_path):
    model,c,modelid=model_from_file(model_file)
    assert modelid==hashlib.sha256(model_file.read_bytes()).digest()
    checkpoint=load_checkpoint(model_file)
    load_weights(model,checkpoint)
    path=tmp_path/'legacy.pt';torch.save({'model':{},'config':{'model':{'message_bits':64}}},path)
    with pytest.raises(ValueError,match='legacy'): load_checkpoint(path)
    del checkpoint['model']['watermark_decoder.head.bias']
    with pytest.raises(ValueError,match='keys'): load_weights(model,checkpoint)
    checkpoint=load_checkpoint(model_file)
    checkpoint['model']['color_decoder.dct.kernels'][0,0,0,0]+=1
    with pytest.raises(ValueError,match='DCT buffer'): load_weights(model,checkpoint)


def test_decoder_tail_interfaces_match_forward():
    model=DualPayloadSystem().eval()
    gray=torch.rand(1,1,256,256)
    with torch.no_grad():
        dc=model.color_decoder;dw=model.watermark_decoder
        torch.testing.assert_close(dc(gray)['rgb'],dc.forward_from_coefficients(dc.dct(gray))['rgb'],atol=0,rtol=0)
        torch.testing.assert_close(dw(gray),dw.forward_from_coefficients(dw.dct.watermark(gray)),atol=0,rtol=0)
        assert dw(gray).shape==(1,256)


def test_protocol_interfaces_signature_order_and_no_updates(model_file):
    engine=ProtocolV1(model_file)
    c=config_for('protocol_eval');apply_training_stage(engine.model,c)
    with pytest.raises(ValueError,match='optimizer'): build_optimizer(engine.model,c)
    assert not engine.model.training
    assert all(not p.requires_grad for p in engine.model.parameters())
    before={n:v.clone() for n,v in engine.model.state_dict().items()}
    kc,km,token=b'c'*32,b'm'*32,b't'*16
    sk=Ed25519PrivateKey.generate()
    rgb=torch.rand(1,3,256,256,requires_grad=True)
    sent={}
    handle=engine.model.watermark_encoder.register_forward_hook(
        lambda _,args,out: sent.update(carrier=out["carrier"].clone()))
    data=engine.publish(rgb,token,kc,km,sk,c['protocol'])
    handle.remove()
    verified=verify_package(data,sk.public_key(),expected_model_id=engine.model_id)
    # Assert that the original full forward methods are never re-entered.
    with patch.object(engine.model.color_decoder,'forward',side_effect=AssertionError('redundant DCT')):
        restored=engine.recover_color(data,kc,sk.public_key())
    assert restored.shape==(1,3,256,256) and not restored.requires_grad
    with fp32_inference(engine.device):
        baseline_rgb=engine.model.color_decoder(sent["carrier"])["rgb"]
    torch.testing.assert_close(restored,baseline_rgb,atol=2e-5,rtol=2e-5)
    with fp32_inference(engine.device), patch.object(engine.model.watermark_decoder,'forward',side_effect=AssertionError('redundant DCT')):
        logits=engine._patient_logits(verified,km)
    assert logits.shape==(256,) and not logits.requires_grad
    bad=bytearray(data);bad[1176]^=1
    with patch.object(engine.model.color_decoder.dct,'forward',side_effect=AssertionError('must authenticate first')):
        with pytest.raises(InvalidSignature): engine.recover_color(bytes(bad),kc,sk.public_key())
    # The API needs only its own branch key. Other branch mutation does not affect input features.
    h=verified.header
    with fp32_inference(engine.device):
        cg=engine.model.color_decoder.dct(verified.gray)
        cfeatures=inverse_branch(cg[:,list(CHANNELS[COLOR])],permutation_key(kc,h.nc,COLOR),h.image_id,COLOR,verified.raw.selectors)
        captured=[]
        handle=engine.model.color_decoder.structure_stem.register_forward_pre_hook(lambda _,args:captured.append(args[0]))
        engine.recover_color(data,kc,sk.public_key());handle.remove()
        expected=engine.model.color_decoder.dct.inverse(cg*engine.model.color_decoder.dct.mask_0)
        torch.testing.assert_close(captured[0],expected,atol=0,rtol=0)
        assert cfeatures.shape==(1,39,32,32)
    assert all(p.grad is None for p in engine.model.parameters()) and rgb.grad is None
    for n,v in before.items(): assert torch.equal(v,engine.model.state_dict()[n])


def test_authenticated_patient_release_with_controlled_decoder(model_file):
    """Integration oracle for payload plumbing, explicitly NOT learned decoding evidence."""
    engine=ProtocolV1(model_file)
    sk=Ed25519PrivateKey.generate();km=b'm'*32;token=b't'*16
    h=Header(b'i'*16,engine.model_id,1.,1.,2,2,b'c'*16,b'm'*16,b'n'*12)
    payload=encrypt_patient(token,km,h.nm,h.ngcm,h.encode())
    logits=(payload_to_bits(payload)*2-1).unsqueeze(0)
    data=encode_package(h,bytes(1024),torch.rand(1,1,256,256),sk)
    with patch.object(engine.model.watermark_decoder,'forward_from_coefficients',return_value=logits) as decoder:
        assert engine.recover_patient(data,km,sk.public_key())==token
        assert decoder.call_args.args[0].shape==(1,9,32,32)
        with pytest.raises(InvalidTag): engine.recover_patient(data,b'x'*32,sk.public_key())
    altered=logits.clone();altered[0,0]*=-1
    with patch.object(engine.model.watermark_decoder,'forward_from_coefficients',return_value=altered):
        with pytest.raises(InvalidTag): engine.recover_patient(data,km,sk.public_key())
    damaged=bytearray(data);damaged[152]^=1
    with patch.object(engine.model.watermark_decoder,'forward_from_coefficients',side_effect=AssertionError('no decode')):
        with pytest.raises(InvalidSignature):engine.recover_patient(bytes(damaged),km,sk.public_key())


def test_protocol_rejects_model_mutation(model_file):
    engine=ProtocolV1(model_file)
    with torch.no_grad(): next(engine.model.parameters()).add_(1)
    with pytest.raises(ValueError,match='changed'):engine._check_model()


def test_cli_synthetic_training_and_protocol_eval(tmp_path):
    # One synthetic optimization step exercises the real loop, never PAD-UFES-20.
    c=config_for('joint_256');c['data']['batch_size']=1;c['train']['epochs']=1
    path=tmp_path/'joint.json';path.write_text(json.dumps(c))
    run=tmp_path/'train'
    with patch('dual_payload.training.DualPayloadSystem',wraps=DualPayloadSystem) as constructor, \
         patch('torch.nn.Module.load_state_dict',side_effect=AssertionError('training must start fresh')):
        train_main(['--config',str(path),'--smoke','--device','cpu','--output-dir',str(run),'--max-steps','1'])
    assert constructor.call_count==1
    state=load_checkpoint(run/'last.pt')
    assert state['global_step']==1 and state['synthetic'] and state['config']['model']['message_bits']==256
    assert state['epoch']==0 and state['config']['train']['stage']=='joint_256'
    assert state['overfit8'] is False
    history=(run/'metrics.jsonl').read_text().splitlines()
    assert len(history)==1
    epoch_metrics=json.loads(history[0])
    assert epoch_metrics['epoch']==state['epoch'] and epoch_metrics['global_step']==1
    assert epoch_metrics['train_samples']==1 and epoch_metrics['mode']=='joint_256'
    assert set(epoch_metrics['train_loss'])=={'rgb','chroma','luma','message','carrier','range','total'}
    assert epoch_metrics['validation']==state['validation']==json.loads((run/'validation.json').read_text())
    assert all(v['step'].item()==1 for v in state['optimizer']['state'].values())
    ec=config_for('protocol_eval');evalpath=tmp_path/'eval.json';evalpath.write_text(json.dumps(ec))
    args=['--config',str(evalpath),'--checkpoint',str(run/'last.pt'),'--smoke','--device','cpu',
          '--output-dir',str(tmp_path/'eval')]
    for name,length in [('kc',32),('km',32),('signing-key',32),('token',16)]:
        keypath=tmp_path/(name+'.bin');keypath.write_bytes(bytes(range(length)));args += ['--'+name+'-file',str(keypath)]
    with patch("dual_payload.protocol.encrypt_patient", wraps=encrypt_patient) as encryption, \
         patch("dual_payload.protocol.decrypt_patient", wraps=decrypt_patient) as decryption:
        evaluate_main(args)
        assert encryption.call_count == 1  # BER must not cause nonce reuse.
        assert decryption.call_count == 1  # Real logits reach GCM; InvalidTag is allowed.
    report=json.loads((tmp_path/'eval'/'report.json').read_text())
    assert report['stage']=='protocol_eval' and report['synthetic']
    assert len((tmp_path/'eval'/'000000.dpw').read_bytes())==263384
    assert isinstance(report['images'][0]['patient_authenticated'],bool)
    with pytest.raises(ValueError,match='prohibited'):
        train_main(['--config',str(evalpath),'--smoke','--output-dir',str(tmp_path/'forbidden')])


def test_no_implicit_training():
    with pytest.raises(SystemExit):train_main([])


def test_protocol_config_fields_must_be_explicit(tmp_path):
    c=config_for('joint_256');del c['protocol']['candidate_count']
    p=tmp_path/'c.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='Explicit protocol'):load_config(p)


def test_checkpoint_rejects_tensor_dtype_before_copy(model_file):
    model=DualPayloadSystem();checkpoint=load_checkpoint(model_file)
    checkpoint['model']['watermark_decoder.head.weight']=checkpoint['model']['watermark_decoder.head.weight'].half()
    before=model.watermark_decoder.head.weight.detach().clone()
    with pytest.raises(ValueError,match='dtype'):load_weights(model,checkpoint)
    assert torch.equal(before,model.watermark_decoder.head.weight)
