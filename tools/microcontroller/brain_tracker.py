# brain_tracker.py — ESP32 + SSD1306 OLED "fly brain tracker".
#
# Connects to Wi-Fi, periodically fetches a random neuron's stats from the
# Virtual Fly Brain API, and shows them on a 128x64 SSD1306 OLED.
#
# Files on the board (Thonny / ampy / mpremote):
#   main.py      <- this file
#   lib/ssd1306.py  (MicroPython SSD1306 driver)
#
# Wiring (I2C, common ESP32 devkit):
#   OLED SDA -> GPIO 21, SCL -> GPIO 22, VCC -> 3V3, GND -> GND
#
# Honesty note: this shows real neuron metadata from VFB's API as a fun desk
# widget. It is not a live readout of a biological fly brain.

import machine
import network
import time
import urequests
import ujson

# ---------------- config ----------------
WIFI_SSID = "your-wifi-ssid"
WIFI_PASS = "your-wifi-pass"

VFB_API = "https://virtualflybrain.org/api/v2/find/neurons"
FETCH_EVERY_S = 30
I2C_SDA = 21
I2C_SCL = 22

# ---------------- display ----------------
i2c = machine.I2C(0, scl=machine.Pin(I2C_SCL), sda=machine.Pin(I2C_SDA), freq=400000)
import ssd1306  # put the driver in lib/ and set sys.path, or vendor it here
oled = ssd1306.SSD1306_I2C(128, 64, i2c)


def banner(line1: str, line2: str = "") -> None:
    oled.fill(0)
    oled.text("fly-brain tracker", 0, 0)
    oled.text(line1[:21], 0, 20)
    oled.text(line2[:21], 0, 34)
    oled.show()


# ---------------- wifi ----------------
def connect_wifi() -> None:
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print("connecting to wifi ...")
        wlan.connect(WIFI_SSID, WIFI_PASS)
        for _ in range(40):
            if wlan.isconnected():
                break
            banner("wifi connecting", "." * (_ % 21))
            time.sleep(0.5)
    if not wlan.isconnected():
        banner("wifi FAILED", "check SSID/pass")
        raise RuntimeError("wifi connect failed")
    banner("wifi OK", wlan.ifconfig()[0])
    print("wifi ok:", wlan.ifconfig()[0])


# ---------------- vfb fetch ----------------
def fetch_random_neuron() -> dict | None:
    """Ask VFB for a neuron entry and return {id, label, note} or None."""
    try:
        url = VFB_API + "?rows=1&random=true"
        r = urequests.get(url, headers={"Accept": "application/json"})
        data = r.json()
        r.close()
        results = data.get("results", []) if isinstance(data, dict) else []
        if not results:
            return None
        n = results[0]
        return {
            "id": n.get("short_form") or n.get("id") or "?",
            "label": n.get("label") or n.get("name") or "?",
            "note": n.get("note") or n.get("classification") or "",
        }
    except Exception as e:  # noqa: BLE001 — board code must never die on a bad response
        print("fetch error:", e)
        return None


def show_neuron(n: dict) -> None:
    oled.fill(0)
    oled.text("fly-brain tracker", 0, 0)
    oled.text(n["id"][:21], 0, 16)
    # wrap the label over two lines
    label = n["label"]
    oled.text(label[:21], 0, 30)
    if len(label) > 21:
        oled.text(label[21:42], 0, 40)
    t = time.localtime()
    oled.text("%02d:%02d:%02d" % (t[3], t[4], t[5]), 0, 54)
    oled.show()


def main() -> None:
    connect_wifi()
    while True:
        n = fetch_random_neuron()
        if n:
            print("neuron:", n["id"], n["label"])
            show_neuron(n)
        else:
            banner("fetch failed", "retrying ...")
        time.sleep(FETCH_EVERY_S)


main()
