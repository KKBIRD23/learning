"""
请求地址:http://10.50.14.218:8080/sms/services/MsgService
或
请求地址:http://10.50.15.9:34005

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

"""
from suds.client import Client

# url = "http://10.50.15.9:34005"
url = "http://www.webxml.com.cn/webservices/qqOnlineWebService.asmx?wsdl"
headers = {"Content-Type": 'application/soap+xml;charset="UTF-8"'}

params_dict = {"PhoneNo": "18983884801", "Content": "testmsg"}

client = Client(url, headers=headers, faults=False, timeout=15)

print(client)

# result = client.service.SendSMS(params_dict)
