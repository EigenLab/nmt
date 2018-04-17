import requests
import json

def send_to_dingding(text, level = None):
    url = "https://oapi.dingtalk.com/robot/send?access_token=15cf72b2aca759d8cd4cdeac22aced0fe360885ce588fc4d0c76c3b64aa34e05"
    HEADERS = {
        "Content-Type": "application/json ;charset=utf-8 "
    }
    String_textMsg = {"msgtype": "text","text": {"content": text}}
    String_textMsg = json.dumps(String_textMsg).replace("</", "<\\/")
    try:
        requests.post(url, data=String_textMsg, headers=HEADERS, timeout=5)
    except:
        print("Fail to send message to dingding")