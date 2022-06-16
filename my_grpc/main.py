from pickle import TRUE
import grpc
import helloworld_pb2
import helloworld_pb2_grpc
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).cuda()

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        text = input()
        input_ids = tokenizer.encode(text, return_tensors="pt").cuda()
        out = model.generate(input_ids.cuda())
        generated_text = list(map(tokenizer.decode, out))[0]
        print(generated_text)
        response = stub.SayHello(helloworld_pb2.HelloRequest(name=generated_text))
        print(response.message)

while True:
    print("i`m working")
    run()