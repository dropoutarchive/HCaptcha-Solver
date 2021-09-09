import asyncio
import base64
import hashlib
import json
import math
import logging
import random
import os
import urllib
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import cvlib as cv
import cv2
import tensorflow as tf

import aiohttp
import tasksio

logging.basicConfig(
    level=logging.INFO,
    format=f"\u001b[36;1m[\u001b[0m%(asctime)s\u001b[36;1m]\u001b[0m -> \u001b[36;1m%(message)s\u001b[0m",
    datefmt="%H:%M:%S",
)

class HCaptcha(object):

    def __init__(self, debug: bool = True):
        self.debug = debug
        self.timeout = aiohttp.ClientTimeout(total=3)
        self.headers = {
            "Authority": "hcaptcha.com",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://assets.hcaptcha.com",
            "Sec-Fetch-Site": "same-site",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Accept-Language": "en-US,en;q=0.9"
        }
        self.answers = {}

    async def _mouse_movement(self):
        movement = []

        for x in range(50, 100):
            x_movement = random.randint(15, 450)
            y_movement = random.randint(15, 450)
            rounded_time = round(datetime.now().timestamp())
            movement.append([x_movement, y_movement, rounded_time])

        return movement

    async def _dehash(self, hash):
        x = "0123456789/:abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        req = hash.split(".")

        req = {
            "header": json.loads(
                base64.b64decode(
                    req[0] +
                    "=======").decode("utf-8")),
            "payload": json.loads(
                base64.b64decode(
                    req[1] +
                    "=======").decode("utf-8")),
            "raw": {
                "header": req[0],
                "payload": req[1],
                "signature": req[2]}}

        def a(r):
            for t in range(len(r) - 1, -1, -1):
                if r[t] < len(x) - 1:
                    r[t] += 1
                    return True
                r[t] = 0
            return False

        def i(r):
            t = ""
            for n in range(len(r)):
                t += x[r[n]]
            return t

        def o(r, e):
            n = e
            hashed = hashlib.sha1(e.encode())
            o = hashed.hexdigest()
            t = hashed.digest()
            e = None
            n = -1
            o = []
            for n in range(n + 1, 8 * len(t)):
                e = t[math.floor(n / 8)] >> n % 8 & 1
                o.append(e)
            a = o[:r]

            def index2(x, y):
                if y in x:
                    return x.index(y)
                return -1
            return 0 == a[0] and index2(a, 1) >= r - 1 or -1 == index2(a, 1)

        def get():
            for e in range(25):
                n = [0 for i in range(e)]
                while a(n):
                    u = req["payload"]["d"] + "::" + i(n)
                    if o(req["payload"]["s"], u):
                        return i(n)

        result = get()
        hsl = ":".join([
            "1",
            str(req["payload"]["s"]),
            datetime.now().isoformat()[:19]
            .replace("T", "")
            .replace("-", "")
            .replace(":", ""),
            req["payload"]["d"],
            "",
            result
        ])
        return hsl

    async def _get_site_config(self, host, site_key):
        try:
            async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
                async with session.get("https://hcaptcha.com/checksiteconfig?host=%s&sitekey=%s&sc=1&swa=1" % (host, site_key)) as response:
                    json_resp = await response.json()
                    if json_resp["pass"] == True:
                        json_resp["c"]["type"] = "hsl"
                        return json_resp["c"]
                    else:
                        raise Exception("_get_site_config: HCaptcha did not return a valid response")
        except Exception:
            raise Exception("_get_site_config: Unable to send request")

    async def _get_captcha(self, host, site_key, mouse_movement, hash, site_config):
        st = round(datetime.now().timestamp())
        data = urllib.parse.urlencode({
            "host": host,
            "sitekey": site_key,
            "hl": "en",
            "motionData": {
                "mm": mouse_movement,
                "st": st,
                "v": 1,
                "session": [],
                "prev": {
                    "escaped": False,
                    "passed": False,
                    "expiredChallenge": False,
                    "expiredResponse": False
                },
                "topLevel": {
                    "inv": False,
                    "st": st,
                    "sc": {
                        "availWidth": 1366,
                        "availHeight": 742,
                        "width": 1366,
                        "height": 768,
                        "colorDepth": 24,
                        "pixelDepth": 24,
                        "availLeft": 0,
                        "availTop": 26
                    },
                    "plugins": [
                        "wYMmzZs279evXLF",
                        "w48.fPHDBAgwYMmzZMGjRIky58. fv37",
                        "IEiRo0at279. fPHDBgwYs27dOHjx4cO",
                        "KFCBgw48ev37du3b"
                    ]
                }
            },
            "n": hash,
            "c": json.dumps(site_config)
        })

        self.headers["Content-Length"] = str(len(data))
        try:
            async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
                async with session.post("https://hcaptcha.com/getcaptcha?s=%s" % (site_key), data=data) as response:
                    if response.status == 200:
                        return (await response.json())
                    else:
                        raise Exception("_get_captcha: HCaptcha did not return a valid response")
        except Exception:
            raise Exception("_get_captcha: Unable to send request")

    async def _image_recognition(self, image, task_key, object):
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(image) as response:
                    content = response.content.read_nowait()

                nparr = np.frombuffer(content, np.uint8)
                im = cv2.imdecode(nparr, flags=1)
                objects = cv.detect_common_objects(im, confidence=0.8, nms_thresh=1)[1]

                if object.lower() in objects:
                    if self.debug: logging.info("%s.jpg is a %s" % (task_key, object))
                    self.answers[task_key] = "true"
                else:
                    if self.debug: logging.info("%s.jpg is not a %s" % (task_key, object))
                    self.answers[task_key] = "false"
        except Exception:
            return await self._image_recognition(image, task_key, object)

    async def _submit(self, key, host, site_key, site_config, mouse_movement, hash):
        motion_data = {
            "mm": mouse_movement,
            "st": round(datetime.now().timestamp()),
            "v": 1,
            "session": [],
            "prev": {
                "escaped": False,
                "passed": False,
                "expiredChallenge": False,
                "expiredResponse": False
            },
            "topLevel": {
                "inv": False,
                "st": round(datetime.now().timestamp()),
                "sc": {
                    "availWidth": 1366,
                    "availHeight": 742,
                    "width": 1366,
                    "height": 768,
                    "colorDepth": 24,
                    "pixelDepth": 24,
                    "availLeft": 0,
                    "availTop": 26
                },
                "plugins": [
                    "wYMmzZs279evXLF",
                    "w48.fPHDBAgwYMmzZMGjRIky58. fv37",
                    "IEiRo0at279. fPHDBgwYs27dOHjx4cO",
                    "KFCBgw48ev37du3b"
                ]
            }
        }
        data = {
            "answers": self.answers,
            "c": json.dumps(site_config),
            "job_mode": "image_label_binary",
            "motionData": json.dumps(motion_data),
            "n": hash,
            "serverdomain": host,
            "sitekey": site_key,
            "v": "1eed1c2"
        }
        self.headers["Content-Length"] = str(len(data))
        try:
            async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
                async with session.post("https://hcaptcha.com/checkcaptcha/%s" % (key), data=data) as response:
                    json_resp = await response.json()
                    if json_resp["success"]:
                        if json_resp["pass"]:
                            if self.debug: logging.info(json_resp)
                            return {
                                "req": json_resp["c"]["req"],
                                "expiration": json_resp["expiration"],
                                "generated_pass_UUID": json_resp["generated_pass_UUID"]
                            }
                        else:
                            if self.debug: logging.info("Solve process failed")
                            return
                    else:
                        if self.debug: logging.info("Solve process failed")
                        return
        except Exception:
            raise Exception("_submit: Unable to send request")

    async def start(self, host, site_key):
        mouse_movement = await self._mouse_movement()
        if self.debug: logging.info(mouse_movement)
        site_config = await self._get_site_config(host, site_key)
        if self.debug: logging.info(site_config)
        hash = await self._dehash(site_config["req"])
        if self.debug: logging.info(hash)
        captcha = await self._get_captcha(host, site_key, mouse_movement, hash, site_config)
        if self.debug: logging.info(captcha)
        key = captcha["key"]
        hash = await self._dehash(captcha["c"]["req"])
        if self.debug: logging.info(hash)
        captcha["c"]["type"] = "hsl"
        site_config = captcha["c"]
        if self.debug: logging.info(site_config)

        object = captcha["requester_question"]["en"].split(" ")[-1].replace("motorbus", "bus")
        if self.debug: logging.info("Searching for images with a %s" % (object))

        async with tasksio.TaskPool(500) as pool:
            for question in captcha["tasklist"]:
                await pool.put(self._image_recognition(question["datapoint_uri"], question["task_key"], object))

        if self.debug: logging.info(self.answers)
        awnser = await self._submit(key, host, site_key, site_config, mouse_movement, hash)
        if self.debug: logging.info("Result -> %s" % (awnser))
        return awnser

if __name__ == "__main__":
    client = HCaptcha(
        debug=True  
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.start("www.tokyobitcoiner.com", "37f92ac1-4956-457e-83cd-723423af613f"))
