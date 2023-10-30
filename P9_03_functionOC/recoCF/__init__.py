import logging
import azure.functions as func
from .predict import predict_reco_CF
import json


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    userId = req.params.get('userId')
    if not userId:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            userId = req_body.get('userId')

    if userId:
        results = predict_reco_CF(int(userId))
        print("results : ", results)
#       return func.HttpResponse(f"Hello, {userId}. This HTTP triggered function executed successfully.")
#       return func.HttpResponse(f"Hello, {results}. This HTTP triggered function executed successfully.")
        return func.HttpResponse(json.dumps(results))#working for a fixed results
    else:
        return func.HttpResponse("This HTTP triggered function executed successfully. Pass a userId in the query string or in the request body for a personalized response.",
             status_code=200
        )
