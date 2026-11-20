"""Generate a small fictional manual (PX-200 conveyor drive) for CI and the demo.

It has the things real manuals have and naive pipelines trip on: running
headers/footers, numbered headings, fault-code tables and safety warnings.
Real evaluation should use real manuals in data/corpus.

  python scripts/make_sample_manual.py data/sample/px200_manual.pdf
"""

import sys

import pymupdf as fitz

HEADER = "PX-200 Conveyor Drive Unit - Installation and Service Manual - Rev C"

# One list entry per page: (heading level, text) where level 0 is body text.
PAGES = [
    [
        (1, "PX-200 Conveyor Drive Unit"),
        (
            0,
            "Installation, operation and service manual for the PX-200 series, models PX-200-4 (4 kW) "
            "and PX-200-7 (7.5 kW). Keep this manual with the unit. Read the safety chapter before any work.",
        ),
    ],
    [
        (1, "1 Safety"),
        (2, "1.1 Qualified personnel"),
        (
            0,
            "Only electricians trained on variable speed drives may open the terminal cover. "
            "Mechanical work on the gearbox may be done by trained maintenance staff.",
        ),
        (2, "1.2 Residual voltage"),
        (
            0,
            "WARNING: The DC bus capacitors stay charged after power is removed. Wait at least 5 minutes "
            "after isolating mains power and confirm less than 50 V DC between terminals DC+ and DC- "
            "before touching any terminal.",
        ),
        (2, "1.3 Lockout"),
        (
            0,
            "Apply lockout/tagout at the main isolator before mechanical work. The emergency stop "
            "button is not an isolation device.",
        ),
    ],
    [
        (1, "2 Technical data"),
        (2, "2.1 Ratings"),
        (
            0,
            "PX-200-4: rated power 4 kW, rated current 8.6 A, supply 380-480 V AC three phase. "
            "PX-200-7: rated power 7.5 kW, rated current 15.2 A, supply 380-480 V AC three phase. "
            "Overload capacity 150 percent for 60 seconds.",
        ),
        (2, "2.2 Environment"),
        (
            0,
            "Ambient temperature -10 to +40 C without derating. Above 40 C derate output current by "
            "2 percent per degree up to a maximum of 50 C. Protection class IP55. Maximum altitude "
            "1000 m without derating.",
        ),
    ],
    [
        (1, "3 Installation"),
        (2, "3.1 Mounting"),
        (
            0,
            "Mount the unit with the gearbox breather plug at the top. Use four M12 bolts of strength "
            "class 8.8 and tighten them to 79 Nm. Leave 100 mm free space above and below the drive "
            "for cooling air.",
        ),
        (2, "3.2 Shaft alignment"),
        (
            0,
            "Align the output shaft with the conveyor head pulley within 0.1 mm radial offset and "
            "0.05 degrees angular misalignment. Check alignment with a dial indicator after the "
            "mounting bolts are tightened.",
        ),
    ],
    [
        (2, "3.3 Electrical connection"),
        (
            0,
            "Connect mains to terminals L1, L2 and L3 and the motor to U, V and W. Use copper cable "
            "rated 75 C. Recommended fuse: 16 A gG for PX-200-4 and 25 A gG for PX-200-7. Connect the "
            "protective earth before any other conductor. Cable glands are M25 for power and M16 for "
            "control cables.",
        ),
        (2, "3.4 Control terminals"),
        (
            0,
            "Terminal DI1 is start/stop, DI2 is direction reverse, DI3 is fault reset. AI1 accepts "
            "0-10 V speed reference. Relay RO1 opens on any active fault.",
        ),
    ],
    [
        (1, "4 Commissioning"),
        (2, "4.1 Parameter setup"),
        (
            0,
            "Enter motor nameplate data in parameters P1.01 (rated voltage), P1.02 (rated current), "
            "P1.03 (rated speed) and P1.04 (rated frequency). Set acceleration time in P2.10, default "
            "5.0 s, and deceleration time in P2.11, default 5.0 s.",
        ),
        (2, "4.2 Autotune"),
        (
            0,
            "Run the static autotune by setting P1.20 to 1 with the motor stopped. The display shows "
            "TUNE for about 40 seconds. Run the rotating autotune (P1.20 = 2) only with the belt "
            "removed from the conveyor.",
        ),
    ],
    [
        (1, "5 Fault codes"),
        (
            0,
            "E-01 Overcurrent: output current exceeded 220 percent. Check motor cable for short circuit "
            "and increase acceleration time P2.10.",
        ),
        (
            0,
            "E-03 DC bus overvoltage: braking energy too high. Increase deceleration time P2.11 or fit "
            "the optional brake resistor BR-200.",
        ),
        (0, "E-05 Undervoltage: supply below 320 V. Check mains supply and fuses."),
        (
            0,
            "E-07 Motor overtemperature: PTC thermistor tripped. Check cooling fan, clean fins, "
            "reduce load. Reset is possible only after the motor has cooled.",
        ),
        (
            0,
            "E-09 Earth fault: leakage current to earth detected. Measure motor insulation with a "
            "500 V megger; minimum 1 megaohm.",
        ),
        (0, "E-12 Communication loss: fieldbus timeout. Check cable and parameter P5.03 timeout."),
    ],
    [
        (1, "6 Maintenance"),
        (2, "6.1 Schedule"),
        (
            0,
            "Every 3 months: clean cooling fins and check fan operation. Every 12 months: check "
            "terminal tightness and gearbox breather. Every 10,000 operating hours or 2 years, whichever "
            "comes first: change gearbox oil. Every 5 years: replace DC bus capacitors' cooling fan.",
        ),
        (2, "6.2 Gearbox oil"),
        (
            0,
            "Use ISO VG 220 synthetic gear oil. Fill quantity is 1.2 litres for PX-200-4 and 1.8 litres "
            "for PX-200-7. Drain while warm. Oil level must reach the centre of the sight glass.",
        ),
    ],
    [
        (2, "6.3 Bearings"),
        (
            0,
            "Output shaft bearings are sealed and lubricated for life. Replace them if noise or "
            "vibration exceeds 4.5 mm/s RMS measured on the gearbox housing.",
        ),
        (2, "6.4 Spare parts"),
        (
            0,
            "Cooling fan kit: part number PX-FAN-02. Gearbox seal kit: PX-SEAL-11. Brake resistor: "
            "BR-200. Keypad: PX-KP-01.",
        ),
    ],
    [
        (1, "7 Disposal"),
        (
            0,
            "Dispose of electronic parts as electronic waste according to local regulations. "
            "Drain gearbox oil before disposal and hand it to an approved waste oil collector.",
        ),
    ],
]


def build(out_path: str) -> None:
    doc = fitz.open()
    total = len(PAGES)
    for n, content in enumerate(PAGES, start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 30), HEADER, fontsize=8, fontname="helv")
        page.insert_text((270, 815), f"Page {n} of {total}", fontsize=8, fontname="helv")
        y = 70
        for level, text in content:
            if level:
                size = 16 if level == 1 else 12.5
                y += 8
                page.insert_text((50, y + size), text, fontsize=size, fontname="hebo")
                y += size + 10
            else:
                rect = fitz.Rect(50, y, 545, 780)
                rc = page.insert_textbox(rect, text, fontsize=10, fontname="helv")
                used = rect.height - rc
                y += used + 10
    doc.save(out_path)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/sample/px200_manual.pdf")
