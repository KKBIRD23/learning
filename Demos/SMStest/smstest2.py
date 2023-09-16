"""
接口地址:http://10.50.14.218:8080/sms/services/MsgService
请求参数:
PhoneNo  电话列表，用“;”分隔
Content 短信内容
例:
<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" xmlns:soapenc="http://schemas.xmlsoap.org/soap/encoding/" xmlns:tns="http://tempuri.org/" xmlns:types="http://tempuri.org/encodedTypes" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">
  <soap:Body soap:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
    <tns:SendSMS> <buf-in href="#id1" /></tns:SendSMS>
    <tns:SendSMSRequest id="id1" xsi:type="tns:SendSMSRequest">
      <PhoneNo xsi:type="xsd:string">18888888888</PhoneNo>
      <Content xsi:type="xsd:string">短信内容</Content>
    </tns:SendSMSRequest>
  </soap:Body>
</soap:Envelope>

使用urllib调用wsdl接口发送短信
"""

import urllib.request
import urllib.parse
import json


def send_sms(phone_list, content):
    url = "http://10.50.14.218:8080/sms/services/MsgService?wsdl"
    data = {
        "PhoneNo": phone_list,
        "Content": content
    }
    data = json.dumps(data)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    res = urllib.request.urlopen(req)
    print(res.read().decode('utf-8'))


if __name__ == '__main__':
    phone_list = ['18983884801']
    content = '测试短信'
    send_sms(phone_list, content)