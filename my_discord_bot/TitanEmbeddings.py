import json
import boto3
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION")

aws_client = boto3.client(
    "bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

class TitanEmbeddings(object):
    accept = "application/json"
    content_type = "application/json"
    
    def __init__(self, model_id="amazon.titan-embed-text-v2:0", boto3_client=None, region_name='us-west-1'):
        
        if boto3_client:
            self.bedrock_boto3 = boto3_client
        else:
            # self.bedrock_boto3 = boto3.client(service_name='bedrock-runtime')
            self.bedrock_boto3 = boto3.client(
                service_name='bedrock-runtime', 
                region_name=region_name, 
            )
        self.model_id = model_id

    def __call__(self, text, dimensions, normalize=True):
        """
        Returns Titan Embeddings

        Args:
            text (str): text to embed
            dimensions (int): Number of output dimensions.
            normalize (bool): Whether to return the normalized embedding or not.

        Return:
            List[float]: Embedding
            
        """

        body = json.dumps({
            "inputText": text,
            "dimensions": dimensions,
            "normalize": normalize
        })

        response = self.bedrock_boto3.invoke_model(
            body=body, modelId=self.model_id, accept=self.accept, contentType=self.content_type
        )

        response_body = json.loads(response.get('body').read())

        return response_body['embedding']


def generate_titan_vector_embedding(text):
    bedrock_embeddings = TitanEmbeddings(model_id="amazon.titan-embed-text-v2:0", boto3_client=aws_client)
    
    modelId = "amazon.titan-embed-text-v2:0"
    accept = "application/json"
    contentType = "application/json"

    body = json.dumps({
        "inputText": text
    })

    response = aws_client.invoke_model(
        body=body,
        modelId=modelId,
        accept=accept,
        contentType=contentType
    )
    
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get("embedding")
    return np.array(embedding)

   