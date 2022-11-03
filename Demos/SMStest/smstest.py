# -*- coding: utf-8 -*-

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
import zeep

# from suds import client

# wsdl = "http://10.50.15.9:34005"
wsdl = "http://10.50.14.218:8080/sms/services/MsgService?wsdl"

client = zeep.Client(wsdl)
# client = client.Client(wsdl)

msg = {"PhoneNo": "18983884801", "Content": "test123"}
# msg = {"PhoneNo": "18983884801", "Content": "test123"}

# print(client.service.doDefine('msg'))
print(client.service.doTemplate('msg'))
