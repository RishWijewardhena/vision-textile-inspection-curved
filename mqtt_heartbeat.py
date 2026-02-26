import ssl
import time
import threading
import paho.mqtt.client as mqtt


class MqttHeartbeat(threading.Thread):
    def __init__(self, broker, port, username, password, topic, interval_sec=2.0, tls_insecure=False):
        super().__init__(daemon=True)
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.topic = topic
        self.interval_sec = interval_sec
        self.tls_insecure = tls_insecure

        self._stop_event = threading.Event()

        self.client = mqtt.Client(client_id=f"{topic.replace('/', '_')}_hb")
        self.client.username_pw_set(self.username, self.password)

        # TLS for 8883
        self.client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
        if self.tls_insecure:
            self.client.tls_insecure_set(True)

        # Optional: “offline” if unexpected disconnect (remove if backend rejects it)
        # self.client.will_set(self.topic, payload="off", qos=0, retain=False)

        self.client.reconnect_delay_set(min_delay=1, max_delay=10)

    def run(self):
        self.client.connect(self.broker, self.port, keepalive=30)
        self.client.loop_start()

        try:
            while not self._stop_event.is_set():
                self.client.publish(self.topic, payload="on", qos=0, retain=False)
                time.sleep(self.interval_sec)
        finally:
            self.client.loop_stop()
            self.client.disconnect()

    def stop(self):
        self._stop_event.set()

