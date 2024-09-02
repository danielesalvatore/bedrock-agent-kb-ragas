# A CDK Project for creating an amazon bedrock agent sandbox 

The project is a simple example of a RAG system using Bedrock and OpenSearch Serverless. It includes a simple chatbot that uses a knowledge base to answer questions.
AWS services used:
- Bedrock Agent (exposed via API Gateway + lambda) that uses Claude 3 Haiku (`anthropic.claude-3-haiku-20240307-v1:0`)
- OpenSearch Serverless and Amazon Bedrock Knowledge Base, that uses Amazon Titan Embeddings (`amazon.titan-embed-text-v1`)

The knowledge base is created using a corpus of local documents, located in the `assets/kb-docs` directory, that are loaded into an OpenSearch Serverless collection. The documents are then embedded using the Amazon Titan Embeddings model and loaded into the knowledge base.

<!--     -->

## Deployment Instructions

### Pre-req setup steps

**Before Use:** Review the latest supported regions for Amazon Bedrock. The selected region will need to support Claude Haiku for this deployment to work.

Create the virtual environment within the root of this project using this command

```
python3 -m venv .venv
```

After the init process completes and the virtualenv is created, you can use the following step to activate your virtualenv.

```
source .venv/bin/activate
```

Once the virtualenv is activated, you can install the required dependencies.

```
pip install -r requirements.txt
```

If you have not used CDK before you will need to [install the CDK CLI](https://docs.aws.amazon.com/cdk/v2/guide/cli.html).

If CDK has not been used in the account or region before you must bootstrap it using the following command.

```
cdk bootstrap
```

**Deploy** 

You can use CDK context arguments to make deployment choices and modify the resulting cloud resources. Please take the time to review the available context options before deployment.

Deploy the application without passing in any arguments by running the below command.  
Please pass in the profile and region as arguments as needed. 
Please make sure that the region supports Bedrock and has the `anthropic.claude-3-haiku-20240307-v1:0` and `amazon.titan-embed-text-v1`models active. 

```
cdk deploy --all (--profile <profile> --region <region>)
```

## Ragas Instructions

Access the `ragas` folder and run the following command to evaluate the knowledge base:

```
python main.py
```

This will run the ragas evaluation and print the results to the console.