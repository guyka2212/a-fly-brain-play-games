# microcontroller — ESP32 fly-brain desk tracker

A MicroPython script for an **ESP32 + 128×64 SSD1306 OLED** that connects to
Wi-Fi and periodically shows a random *Drosophila* neuron's stats, fetched live
from the **Virtual Fly Brain** API. A desk widget for the project.

## Hardware

- Any ESP32 devkit (WROOM/WROVER)
- 0.96" SSD1306 OLED, I2C (128×64)

Wiring:

```
OLED SDA -> GPIO 21
OLED SCL -> GPIO 22
OLED VCC -> 3V3
OLED GND -> GND
```

## Install

1. Flash MicroPython: <https://micropython.org/download/esp32/> (use esptool or
   the Thonny flasher).
2. Get the SSD1306 driver onto the board at `lib/ssd1306.py`:
   [micropython/ssd1306.py](https://github.com/micropython/micropython-lib/blob/master/micropython/drivers/display/ssd1306/ssd1306.py)
   (via `mpremote cp ssd1306.py :lib/` or Thonny drag-and-drop).
3. Copy `brain_tracker.py` to the board as `main.py` (or `boot.py` + import).
4. Edit the config block at the top:

```python
WIFI_SSID = "your-wifi-ssid"
WIFI_PASS = "your-wifi-pass"
FETCH_EVERY_S = 30
```

5. Reset the board. The OLED walks through: Wi-Fi connect → fetch → neuron id +
   name, refreshed every 30 s.

## Notes & honesty

- The VFB "random neuron" endpoint shape can change; the board never crashes on
  a bad response — it shows `fetch failed` and retries.
- This displays **real neuron metadata** as a fun desk companion. It is not a
  live readout of a biological brain, and it is independent of the game agent.
- If Wi-Fi is unavailable, the tracker is useless by design (it fetches live);
  use `tools/neuron-fetch/build_curated.py` output if you'd rather build an
  offline variant that cycles through `shared/connectome-data.json` entries.
