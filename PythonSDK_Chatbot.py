import json
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


authenticator = IAMAuthenticator('puP89CTBEWN2-bguNU-co8ESIAcSXgRHl51Uh06_ZPnC')#watson assistant apikey
assistant = AssistantV2(
    version='2021-06-14',
    authenticator = authenticator
)

assistant.set_service_url('https://api.eu-gb.assistant.watson.cloud.ibm.com')#location url
response = assistant.create_session(
        assistant_id='d0cbd1ea-cd4e-4709-bbee-9a7fb3c68f05'#assistant id
    ).get_result()
session_id = response
session_id = session_id["session_id"]
print(type(session_id))
print(session_id)

while True:
    input_text = input("enter the text")
    
    response = assistant.message(
        assistant_id='d0cbd1ea-cd4e-4709-bbee-9a7fb3c68f05',
        session_id=session_id,
        input={
            'message_type': 'text',
            'text': input_text
        }
    ).get_result()
    print(response)
    print(response["output"]["generic"][0]["text"])