import openai

#INPUT
openai.api_key = 'YOUR OPENAI API KEY'

file_path = 'smartphones.xlsx'

def get_command_prompt(description, json):
	return f"Mając opis smartfona ze strony internetowej, oraz wzór odpowiedzi w formacie json, odpowiedz w formacie json (według zadanego wzoru) uzupełniając wartości zmiennych faktycznymi danymi z opisu smartfona. Odpowiedzi mają być krótkie - dane/parametry, nie całym zdaniem. Pisz po polsku. Jeśli nie można stwierdzić czegoś gdyż brakuje danych w opisie, wpisz '???'. Opis smartfona: {description}\n\n Template json do odpowiedzi: {json}. Odpowiedz TYLKO dokumentem json, bez wyjaśnień."

description = """
Xiaomi Poco X5 Pro
Xiaomi Poco X5 Pro
MORE PICTURES
Released 2023, February 07
181g, 7.9mm thickness
Android 12, MIUI 14 for POCO
128GB/256GB storage, no card slot
27%
3,582,319 HITS
346
BECOME A FAN
6.67"
1080x2400 pixels
108MP
2160p
6/8GB RAM
Snapdragon 778G 5G
5000mAh
Li-Po
REVIEW
PRICES
PICTURES
COMPARE
OPINIONS

NETWORK	Technology	
GSM / HSPA / LTE / 5G
2G bands	GSM 850 / 900 / 1800 / 1900 - SIM 1 & SIM 2
3G bands	HSDPA 800 / 850 / 900 / 1700(AWS) / 1900 / 2100 - International
 	HSDPA 850 / 900 / 1700(AWS) / 1900 / 2100 - India
4G bands	1, 2, 3, 4, 5, 7, 8, 20, 28, 38, 40, 41, 66 - International
 	1, 2, 3, 5, 8, 40, 41 - India
5G bands	1, 3, 5, 7, 8, 20, 28, 38, 40, 41, 77, 78 SA/NSA/Sub6 - International
 	1, 3, 5, 8, 28, 40, 78 SA/NSA - India
Speed	HSPA, LTE-A (CA), 5G
LAUNCH	Announced	2023, February 06
Status	Available. Released 2023, February 07
BODY	Dimensions	162.9 x 76 x 7.9 mm (6.41 x 2.99 x 0.31 in)
Weight	181 g (6.38 oz)
Build	Glass front (Gorilla Glass 5), plastic back, plastic frame
SIM	Dual SIM (Nano-SIM, dual stand-by)
 	IP53, dust and splash resistant
DISPLAY	Type	AMOLED, 1B colors, 120Hz, Dolby Vision, HDR10+, 500 nits (typ), 900 nits (HBM)
Size	6.67 inches, 107.4 cm2 (~86.8% screen-to-body ratio)
Resolution	1080 x 2400 pixels, 20:9 ratio (~395 ppi density)
Protection	Corning Gorilla Glass 5
PLATFORM	OS	Android 12, MIUI 14 for POCO
Chipset	Qualcomm SM7325 Snapdragon 778G 5G (6 nm)
CPU	Octa-core (1x2.4 GHz Cortex-A78 & 3x2.2 GHz Cortex-A78 & 4x1.9 GHz Cortex-A55)
GPU	Adreno 642L
MEMORY	Card slot	No
Internal	128GB 6GB RAM, 256GB 8GB RAM
 	UFS 2.2
MAIN CAMERA	Triple	108 MP, f/1.9, (wide), 1/1.52", 0.7µm, PDAF
8 MP, f/2.2, 120˚ (ultrawide), 1/4", 1.12µm
2 MP, f/2.4, (macro)
Features	LED flash, HDR, panorama
Video	4K@30fps, 1080p@30/60/120fps, gyro-EIS
SELFIE CAMERA	Single	16 MP, f/2.4, (wide), 1/3.06" 1.0µm
Features	HDR, panorama
Video	1080p@30/60fps
SOUND	Loudspeaker	Yes, with stereo speakers
3.5mm jack	Yes
 	24-bit/192kHz audio
COMMS	WLAN	Wi-Fi 802.11 a/b/g/n/ac/6, dual-band, Wi-Fi Direct
Wi-Fi 802.11 a/b/g/n/ac, dual-band - India
Bluetooth	5.2 (Intl), 5.1 (India), A2DP, LE
Positioning	GPS, GLONASS, BDS, GALILEO
NFC	Yes (market/region dependent)
Infrared port	Yes
Radio	No
USB	USB Type-C 2.0, OTG
FEATURES	Sensors	Fingerprint (side-mounted), accelerometer, gyro, proximity, compass
BATTERY	Type	Li-Po 5000 mAh, non-removable
Charging	67W wired, PD3.0, QC4, 100% in 45 min (advertised)
5W reverse wired
MISC	Colors	Astral Black, Horizon Blue, Poco Yellow
Models	22101320G, 22101320I
Price	€ 247.99 / $ 245.00 / £ 279.99 / ₹ 19,299
TESTS	Performance	AnTuTu: 531398 (v9)
GeekBench: 2930 (v5.1)
GFXBench: 28fps (ES 3.1 onscreen)
Display	Contrast ratio: Infinite (nominal)
Camera	Photo / Video
Loudspeaker	-24.9 LUFS (Very good)
Battery (old)	
Endurance rating 113h
"""