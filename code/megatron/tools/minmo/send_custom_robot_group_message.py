# !/usr/bin/env python

import argparse
import logging
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests

def setup_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('%(asctime)s %(name)-8s %(levelname)-8s %(message)s [%(filename)s:%(lineno)d]'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def define_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--access_token', dest='access_token', required=True,
        help='机器人webhook的access_token from https://open.dingtalk.com/document/orgapp/obtain-the-webhook-address-of-a-custom-robot '
    )
    parser.add_argument(
        '--secret', dest='secret', required=False, default=None,
        help='secret from https://open.dingtalk.com/document/orgapp/customize-robot-security-settings#title-7fs-kgs-36x'
    )
    parser.add_argument(
        '--userid', dest='userid',
        help='待 @ 的钉钉用户ID，多个用逗号分隔 from https://open.dingtalk.com/document/orgapp/basic-concepts-beta#title-o8w-yj2-t8x '
    )
    parser.add_argument(
        '--at_mobiles', dest='at_mobiles',
        help='待 @ 的手机号，多个用逗号分隔'
    )
    parser.add_argument(
        '--is_at_all', dest='is_at_all', action='store_true',
        help='是否@所有人，指定则为True，不指定为False'
    )
    parser.add_argument(
        '--title', dest='title', default='钉钉',
        help='要发送的消息内容'
    )
    parser.add_argument(
        '--msg', dest='msg', default='钉钉，让进步发生',
        help='要发送的消息内容'
    )
    return parser.parse_args()


def send_custom_robot_group_message(access_token, secret, title, msg, at_user_ids=None, at_mobiles=None, is_at_all=False):
    """
    发送钉钉自定义机器人群消息
    :param access_token: 机器人webhook的access_token
    :param secret: 机器人安全设置的加签secret
    :param msg: 消息内容
    :param at_user_ids: @的用户ID列表
    :param at_mobiles: @的手机号列表
    :param is_at_all: 是否@所有人
    :return: 钉钉API响应
    """
    timestamp = str(round(time.time() * 1000))


    msg = msg.replace("\\n", "\n")


    url = f'https://oapi.dingtalk.com/robot/send?access_token={access_token}&timestamp={timestamp}'

    if secret:
        string_to_sign = f'{timestamp}\n{secret}'
        hmac_code = hmac.new(secret.encode('utf-8'), string_to_sign.encode('utf-8'), digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        url += f'&sign={sign}'

    body = {
        "at": {
            "isAtAll": str(is_at_all).lower(),
            "atUserIds": at_user_ids or [],
            "atMobiles": at_mobiles or []
        },
        "markdown": {
            "title": title,
            "text": msg
        },
        "msgtype": "markdown"
    }
    headers = {'Content-Type': 'application/json'}
    resp = requests.post(url, json=body, headers=headers)
    logging.info("钉钉自定义机器人群消息响应：%s", resp.text)
    return resp.json()


def main():
    options = define_options()
    # 处理 @用户ID
    at_user_ids = []
    if options.userid:
        at_user_ids = [u.strip() for u in options.userid.split(',') if u.strip()]
    # 处理 @手机号
    at_mobiles = []
    if options.at_mobiles:
        at_mobiles = [m.strip() for m in options.at_mobiles.split(',') if m.strip()]
    send_custom_robot_group_message(
        options.access_token,
        options.secret,
        options.title,
        options.msg,
        at_user_ids=at_user_ids,
        at_mobiles=at_mobiles,
        is_at_all=options.is_at_all
    )


if __name__ == '__main__':
    main()