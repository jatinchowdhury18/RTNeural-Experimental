import torch
from torch.utils.data import Dataset
import torch.nn as nn
torch.backends.quantized.engine = 'qnnpack'
import json
from json import JSONEncoder
torch.manual_seed(0)

class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.ao.quantization.QuantStub()

        self.dense1 = nn.Linear(4, 4, bias=False)
        
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        res = x

        x = self.quant(x)
        
        x = self.dense1(x)
        
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        
        x = self.dequant(x)
        return x


#INSTANTIATE MODEL and EVAL MODE
model_fp32 = TestModel()
model_fp32.eval()

#GET CONFIG AND PREPARE
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)

#CREATE INPUT
input_fp32 = torch.Tensor([
    [+1.0, +2.0, +3.0, +4.0],
    [+3.0, +4.0, +5.0, +6.0],
    [-1.0, -2.0, -3.0, -4.0],
    [+1.1, +2.2, +3.3, +4.4],
])
model_fp32_prepared(input_fp32)

#ACTUALLY CONVERT
model_int8 = torch.ao.quantization.convert(model_fp32_prepared, dtype=torch.qint8)


print("float 32 model:", model_fp32)
print("int 8 model:", model_int8)

# run the model, relevant calculations will happen in int8

out_fp32 = model_fp32(input_fp32)
print(out_fp32.detach().numpy())

out_int8 = model_int8(input_fp32)
print(out_int8.detach().numpy())


with open('model_fp32.json', 'w') as json_file:
    json.dump(model_fp32.state_dict(), json_file, cls=EncodeTensor)

# with open('model_int8.json', 'w') as json_file:
#     json.dump(model_int8.state_dict(), json_file, cls=EncodeTensor)

print(model_int8.state_dict()['quant.scale'][0])
print(model_int8.state_dict()['quant.zero_point'][0])
print(model_int8.state_dict()['dense1.scale'])
print(model_int8.state_dict()['dense1.zero_point'])
print(model_int8.state_dict()['dense1._packed_params._packed_params'][0].detach())
print(model_int8.state_dict()['dense1._packed_params._packed_params'][1].detach().numpy())
