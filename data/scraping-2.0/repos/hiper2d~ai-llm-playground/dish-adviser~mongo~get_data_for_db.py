import uuid
from typing import List

from langchain.schema import Document

SUSHI_HAVEN = """Restaurant: SUSHI HAVEN
Description: SUSHI HAVEN is a high-end sushi joint nestled in the heart of Tampa, Florida. With a commitment to serving the freshest sushi and sashimi, our team of experienced chefs handpick the finest fish daily. The restaurant's ambiance radiates tranquility, with bamboo accents, minimalist design, and soft lighting. A selection of sake and Japanese whiskies perfectly complements the dishes. Signature rolls, timeless sashimi platters, and mouthwatering tempuras are a staple, ensuring a blend of traditional and contemporary offerings.

LUNCH MENU:

NIGIRI SELECTIONS:
- MAGURO [ID: 101]: Bluefin tuna, lightly seasoned rice, wasabi - $10, 50 cal.
- HAMACHI [ID: 102]: Yellowtail fish, sushi rice, light soy - $9, 45 cal.
- EBI [ID: 103]: Cooked shrimp, rice, pickled ginger - $8, 40 cal.

CLASSIC ROLLS:
- SPICY TUNA [ID: 104]: Bluefin tuna, spicy mayo, cucumber - $15, 60 cal.
- VEGETABLE ROLL [ID: 105]: Avocado, cucumber, pickled radish, and carrots - $12, 55 cal.
- TEMPURA SHRIMP [ID: 106]: Crispy shrimp, avocado, tobiko, spicy mayo - $17, 70 cal.

BENTO BOXES:
- TERIYAKI CHICKEN [ID: 107]: Grilled chicken with teriyaki sauce, steamed rice, pickled vegetables - $20, 150 cal.
- BEEF SUKIYAKI [ID: 108]: Thinly sliced beef, tofu, glass noodles, assorted vegetables - $22, 160 cal.

DINNER MENU:

SASHIMI PLATTERS:
- CHEF'S CHOICE [ID: 201]: Assorted fish, fresh wasabi, pickled ginger - $50, 80 cal.
- SALMON TRIO [ID: 202]: Three types of salmon cuts, shiso leaf, lemon zest - $40, 70 cal.

SIGNATURE ROLLS:
- DRAGON ROLL [ID: 203]: Eel, avocado, tobiko, dragon sauce - $25, 75 cal.
- PHOENIX ROLL [ID: 204]: Spicy scallop, tempura flakes, cucumber, gold leaf - $28, 78 cal.

MAIN COURSES:
- MISO BLACK COD [ID: 205]: Marinated cod, miso glaze, steamed bok choy - $30, 150 cal.
- WAGYU TEPPANYAKI [ID: 206]: Premium wagyu beef, grilled vegetables, ponzu dipping sauce - $60, 200 cal.
- LOBSTER TEMPURA [ID: 207]: Whole lobster, crispy tempura batter, yuzu mayo - $45, 170 cal."""

NAMI_WAVE = """Restaurant: NAMI WAVE
Description: NAMI WAVE is an ocean-inspired sushi retreat situated in the bustling streets of Tampa, Florida. Known for its artful presentations, the restaurant brings the serenity of the sea to your plate. The decor embodies an underwater theme with hints of coral and marine life. Premium sakes and craft Japanese cocktails further elevate the dining experience. The menu is a rich tapestry of fresh sushi, umami-rich rolls, and tempura delights.

LUNCH MENU:

NIGIRI HIGHLIGHTS:
- AKAMI [ID: 301]: Lean tuna, wasabi, sushi rice - $11, 52 cal.
- SAKE [ID: 302]: Fresh salmon, lightly torched, sushi rice - $10, 47 cal.
- TAMAGO [ID: 303]: Sweet omelette, sushi rice - $7, 38 cal.

ROLL SPECIALS:
- OCEAN BREEZE [ID: 304]: Crab meat, avocado, cucumber, topped with salmon and lemon zest - $16, 62 cal.
- GREEN DRAGON [ID: 305]: Tempura shrimp, avocado, topped with thin jalapeno slices - $14, 60 cal.

DINNER MENU:

SASHIMI SAMPLERS:
- TUNA VARIETY [ID: 306]: Assortment of tuna cuts, garnished with edible flowers - $48, 80 cal.
- RAINBOW PLATE [ID: 307]: Selection of the day's freshest fish, served with a side of ponzu - $45, 75 cal.

SIGNATURE DISHES:
- GOLDEN EEL BOWL [ID: 308]: Grilled eel, sushi rice, gold flakes, sweet soy glaze - $27, 78 cal.
- SCALLOP CARPACCIO [ID: 309]: Thinly sliced scallops, yuzu dressing, microgreens - $25, 70 cal. """

SAKURA_DELIGHT = """
Restaurant: SAKURA DELIGHT
Description: SAKURA DELIGHT is an embodiment of the cherry blossom spirit right in the heart of Tampa. The restaurant ambiance is painted with pink hues and soft, ambient lighting, reflecting the essence of Sakura season. Beyond the aesthetics, the dishes are a symphony of flavors, with an exquisite range of sushi, maki rolls, and Japanese delicacies. Paired with a selection of refined sakes, the experience is nothing short of delightful.

LUNCH MENU:

NIGIRI FAVORITES:
- IKURA [ID: 401]: Salmon roe, sushi rice, seaweed wrap - $12, 55 cal.
- UNAGI [ID: 402]: Grilled eel, sushi rice, eel sauce - $11, 54 cal.

ROLL SELECTION:
- SAKURA ROLL [ID: 403]: Tuna, avocado, topped with seared salmon, sakura-shaped radish - $18, 65 cal.
- MIDNIGHT ROLL [ID: 404]: Eel, tempura flakes, black tobiko, topped with thin lime slices - $17, 63 cal.

DINNER MENU:

SASHIMI DELIGHTS:
- KAMPACHI SLICES [ID: 405]: Amberjack, wasabi yuzu soy, chives - $47, 77 cal.
- TRIPLE TREAT [ID: 406]: Tuna, salmon, yellowtail, fresh wasabi, ginger - $46, 76 cal.

SIGNATURE DISHES:
- WAGYU NIGIRI [ID: 407]: Thinly sliced wagyu, sushi rice, truffle oil, sea salt - $30, 82 cal.
- KING CRAB TEMAKI [ID: 408]: Hand roll with king crab meat, spicy mayo, cucumber sticks - $28, 80 cal.
"""

TIDES_OF_TSUKI = """
Restaurant: TIDES OF TSUKI
Description: TIDES OF TSUKI offers a serene sushi escapade, reminiscent of a peaceful beachside evening in St. Pete. The restaurant captures the charm of the sea with its nautical-themed interior, enhanced with ambient wave sounds and soft lighting. Specializing in expertly crafted sushi, the menu takes patrons on a journey through the oceans, with dishes showcasing the finest fish and seafood selections. Complemented by a diverse sake range, dining at TIDES OF TSUKI is a refreshing experience.

LUNCH MENU:

NIGIRI SELECTIONS:
- TORO [ID: 501]: Fatty tuna belly, sushi rice, light soy - $14, 57 cal.
- BOTAN EBI [ID: 502]: Sweet shrimp, sushi rice, gold leaf - $12, 53 cal.

ROLL SPECIALS:
- WAVE ROLL [ID: 503]: Crab meat, avocado, cucumber, topped with a wave of thinly sliced octopus - $19, 67 cal.
- TIDE TWIST [ID: 504]: Salmon, spicy mayo, scallions, wrapped in soy paper - $18, 65 cal.

DINNER MENU:

SASHIMI PLATES:
- OCEAN'S EMBRACE [ID: 505]: A curated selection of today's fresh catch, served on ice - $55, 82 cal.
- PEARL PLATTER [ID: 506]: Premium cuts of fish, garnished with edible pearls and gold flakes - $60, 85 cal.

SIGNATURE ENTREES:
- LOBSTER URAMAKI [ID: 507]: Lobster meat, avocado, asparagus, topped with caviar - $32, 90 cal.
- TSUKI'S DELIGHT [ID: 508]: Assorted sushi and sashimi, chef's choice, served with miso soup - $50, 200 cal.
"""

SAKURA_BREEZE = """
Restaurant: SAKURA BREEZE
Description: SAKURA BREEZE, located in the heart of St. Pete, is a tranquil sanctuary celebrating the elegance of cherry blossoms. The ambiance is adorned with soft pink hues and cherry blossom motifs, offering a visual treat. Beyond the aesthetics, SAKURA BREEZE promises a tantalizing array of sushi, hand rolls, and traditional Japanese dishes. The drink menu boasts an impressive list of handpicked sakes and signature Japanese cocktails.

LUNCH MENU:

NIGIRI CLASSICS:
- KANPACHI [ID: 601]: Amberjack, wasabi, sushi rice - $13, 56 cal.
- SABA [ID: 602]: Mackerel, sushi rice, pickled ginger - $11, 52 cal.

ROLL FAVORITES:
- BLOSSOM ROLL [ID: 603]: Tuna, salmon, yellowtail, avocado, wrapped in pink soy paper - $20, 68 cal.
- BREEZE ROLL [ID: 604]: Eel, tempura flakes, cucumber, drizzled with eel sauce - $19, 66 cal.

DINNER MENU:

SASHIMI VARIETIES:
- SUNSET PLATTER [ID: 605]: Assortment of sashimi cuts, served with a side of fresh wasabi and pickled ginger - $58, 84 cal.
- MOONLIGHT SET [ID: 606]: A curated selection of nigiri and sashimi, paired with a bowl of miso soup - $52, 195 cal.

SIGNATURE DISHES:
- CHIRASHI BOWL [ID: 607]: Sashimi cuts over seasoned sushi rice, garnished with pickles and seaweed - $35, 170 cal.
- SAKURA SPECIAL [ID: 608]: A hand-picked assortment of the chef's favorite sushi and rolls, served with a side of edamame - $48, 210 cal.
"""


CASA_DEL_SOL = """
Restaurant: CASA DEL SOL
Description: CASA DEL SOL is a vibrant Mexican eatery located in the bustling heart of Tampa, Florida. Celebrating the rich flavors of Mexico, the restaurant offers a blend of traditional and modern dishes. With its colorful murals, rustic wooden furniture, and ambient mariachi music, the ambiance transports you straight to a Mexican fiesta. A curated selection of tequilas and mezcales complements the spicy and savory dishes.

LUNCH MENU:

TACO VARIETIES:

CARNE ASADA [ID: 701]: Grilled steak, pico de gallo, guacamole, and cilantro - $10, 300 cal.
AL PASTOR [ID: 702]: Marinated pork, pineapple, red onions, and salsa verde - $9, 280 cal.
RAJAS CON CREMA [ID: 703]: Roasted poblano peppers, cream, corn, and cheese - $8, 250 cal.
ENCHILADA SELECTIONS:

POLLO [ID: 704]: Shredded chicken, green tomatillo sauce, cheese, and crema - $14, 320 cal.
VEGETALES [ID: 705]: Seasonal veggies, red enchilada sauce, cheese, and avocado - $13, 290 cal.
DINNER MENU:

MOLE SPECIALS:

MOLE POBLANO [ID: 706]: Chicken in a rich chocolate and chili sauce, sesame seeds, rice - $20, 350 cal.
MOLE VERDE [ID: 707]: Pork in a green pumpkin seed and chili sauce, served with beans - $22, 370 cal.
SIGNATURE DISHES:

TAMPICO STEAK [ID: 708]: Grilled steak topped with a spicy tomato and jalapeno sauce, guacamole - $28, 400 cal.
CAMARONES A LA DIABLA [ID: 709]: Shrimp in a fiery chili sauce, served with white rice - $26, 380 cal.
"""


MEXICALI_BLISS = """
Restaurant: MEXICALI BLISS
Description: MEXICALI BLISS offers a contemporary twist to classic Mexican cuisine right in Tampa, Florida. The interior is a blend of modern design with traditional Mexican art, creating a cozy yet upscale ambiance. With an extensive list of craft cocktails, wines, and beers, the drink menu is as impressive as the food offerings.

LUNCH MENU:

TOSTADA FAVORITES:

TINGA [ID: 801]: Shredded chicken in a chipotle sauce, lettuce, and crema on a crispy tortilla - $10, 290 cal.
CARNITAS [ID: 802]: Slow-cooked pork, refried beans, salsa, and cheese on a crispy tortilla - $11, 310 cal.
QUESADILLA VARIETIES:

HUITLACOCHE [ID: 803]: Corn mushroom, Oaxaca cheese, and epazote - $12, 275 cal.
CHORIZO [ID: 804]: Spicy Mexican sausage, cheese, and pico de gallo - $13, 320 cal.
DINNER MENU:

TAMALE SPECIALS:

POLLO CON MOLE [ID: 805]: Chicken tamales with rich mole sauce - $18, 340 cal.
RAJAS Y QUESO [ID: 806]: Poblano pepper and cheese tamales - $16, 300 cal.
SIGNATURE DISHES:

PESCADO ZARANDEADO [ID: 807]: Grilled fish marinated in a chili and garlic sauce, served with sautéed veggies - $27, 360 cal.
BARBACOA DE BORREGO [ID: 808]: Slow-cooked lamb in agave leaves, served with consommé and tortillas - $29, 410 cal.
"""


FIESTA_VERDE = """
Restaurant: FIESTA VERDE
Description: FIESTA VERDE celebrates the lush flavors of Mexico's green landscapes. Situated in Tampa, Florida, the restaurant showcases a variety of vegetarian and vegan Mexican dishes. The ambiance is fresh and organic, with hanging plants and green-tinted decor, resonating with the eatery's theme. A wide range of non-alcoholic beverages and herbal teas complements the menu.

LUNCH MENU:

TACO SELECTIONS:

NOPALES [ID: 901]: Grilled cactus, salsa verde, and vegan cheese - $9, 240 cal.
MUSHROOM & CORN [ID: 902]: Sauteed mushrooms, corn, and avocado - $8, 230 cal.
ENCHILADA VARIETIES:

VERDURAS [ID: 903]: Mixed veggies, green enchilada sauce, vegan cheese, and avocado - $13, 260 cal.
BEAN & KALE [ID: 904]: Black beans, kale, red enchilada sauce, and vegan sour cream - $12, 250 cal.
DINNER MENU:

CHILE RELLENO SPECIALS:

QUINOA & VEGGIE [ID: 905]: Poblano pepper stuffed with quinoa and veggies, topped with tomato sauce - $18, 310 cal.
CASHEW CHEESE [ID: 906]: Poblano pepper filled with cashew cheese, topped with green salsa - $19, 325 cal.
SIGNATURE DISHES:

VEGAN POZOLE [ID: 907]: Hominy soup with mushrooms and radishes, served with tortilla chips - $20, 290 cal.
JACKFRUIT CARNITAS [ID: 908]: Jackfruit cooked in a spicy adobo sauce, served with tortillas - $22, 300 cal.
"""


CANTINA_ROJA = """
Restaurant: CANTINA ROJA
Description: CANTINA ROJA brings the fiery spirit of Mexico to the heart of Tampa, Florida. With red-tinted lighting, rustic decor, and vibrant murals, the restaurant exudes warmth and passion. A plethora of tequilas and spicy cocktails complements the hot and zesty dishes on offer.

LUNCH MENU:

TORTA FAVORITES:

MILANESA [ID: 1001]: Breaded chicken, beans, lettuce, avocado, and mayo in a toasted roll - $12, 350 cal.
CUBANA [ID: 1002]: Ham, pork, cheese, jalapenos, and mustard in a toasted roll - $13, 375 cal.
CHILAQUILES VARIETIES:

VERDES [ID: 1003]: Tortilla chips cooked in green salsa, topped with crema, cheese, and eggs - $11, 320 cal.
ROJOS [ID: 1004]: Tortilla chips in red salsa, topped with pulled chicken, onions, and cheese - $12, 340 cal.
DINNER MENU:

FAJITA SPECIALS:

CAMARON [ID: 1005]: Shrimp fajitas with bell peppers, onions, and tomatoes - $24, 380 cal.
VEGETALES [ID: 1006]: Veggie fajitas with zucchini, bell peppers, mushrooms, and onions - $20, 300 cal.

SIGNATURE DISHES:

COCHINITA PIBIL [ID: 1007]: Slow-cooked pork in achiote and citrus, served with pickled red onions - $26, 390 cal.
CHILES EN NOGADA [ID: 1008]: Poblano peppers stuffed with ground meat, fruits, and spices, topped with walnut sauce and pomegranate seeds - $28, 420 cal.
"""


PUEBLA_PLAZA = """
Restaurant: PUEBLA PLAZA
Description: PUEBLA PLAZA pays homage to the culinary traditions of Puebla, Mexico. Nestled in Tampa, Florida, the restaurant showcases dishes known for their rich and diverse flavors. The ambiance is reminiscent of a Mexican plaza, with colorful tiles, fountains, and ambient folk music. The bar boasts an impressive selection of agave spirits and traditional pulques.

LUNCH MENU:

TOSTADA VARIETIES:

CEVICHE [ID: 1101]: Fresh fish marinated in lime, with tomatoes, onions, and cilantro on a crispy tostada - $12, 280 cal.
BEAN & CHEESE [ID: 1102]: Refried beans, cheese, lettuce, and salsa on a crispy tostada - $10, 260 cal.
MOLLETE SELECTIONS:

CHORIZO [ID: 1103]: Open-faced sandwich with beans, chorizo, and melted cheese - $11, 330 cal.
AGUACATE [ID: 1104]: Open-faced sandwich with beans, avocado, and melted cheese - $10, 310 cal.
DINNER MENU:

MOLE PUEBLANO SPECIALS:

TURKEY IN MOLE [ID: 1105]: Slow-cooked turkey in a rich and spicy mole sauce, served with rice - $25, 390 cal.
CHICKEN IN MOLE [ID: 1106]: Chicken thighs in a traditional mole sauce, served with beans - $23, 370 cal.
SIGNATURE DISHES:

CEMITA [ID: 1107]: Puebla-style sandwich with breaded meat, avocado, Oaxaca cheese, and chipotle - $18, 410 cal.
TETELA [ID: 1108]: Triangular corn masa pocket filled with beans and cheese, topped with green salsa - $20, 350 cal.
"""


def get_asian_documents() -> List[Document]:
    return [
        Document(
            page_content=SUSHI_HAVEN,
            name="SUSHI HAVEN",
            metadata={
                "id": str(uuid.uuid4()),
                "location": {
                      "type": "Point",
                      "coordinates": [-82.35658898618775, 28.155384135951152]  # [longitude, latitude]
                }
            }
        ),
        Document(
            page_content=NAMI_WAVE,
            name="NAMI WAVE",
            metadata={
                "id": str(uuid.uuid4()),
                "location": {
                    "type": "Point",
                    "coordinates": [-82.3267615726919, 28.189450088627343]  # [longitude, latitude]
                }
            }
        ),
        Document(
            page_content=SAKURA_DELIGHT,
            name="SAKURA DELIGHT",
            metadata={
                "id": str(uuid.uuid4()),
                "location": {
                    "type": "Point",
                    "coordinates": [-82.32653074386218, 28.14248959300198]  # [longitude, latitude]
                }
            }
        ),
        Document(
            page_content=TIDES_OF_TSUKI,
            name="TIDES OF TSUKI",
            metadata={
                "id": str(uuid.uuid4()),
                "location": {
                    "type": "Point",
                    "coordinates": [-82.3422987861816, 28.198898047406693]  # [longitude, latitude]
                }
            }
        ),
        Document(
            page_content=SAKURA_BREEZE,
            name="SAKURA BREEZE",
            metadata={
                "id": str(uuid.uuid4()),
                "location": {
                    "type": "Point",
                    "coordinates": [-82.35091428618331, 28.187044244284305]  # [longitude, latitude]
                }
            }
        ),
    ]


def get_mexican_documents() -> List[Document]:
    return [
        Document(
            page_content=CASA_DEL_SOL,
            name="CASA DEL SOL",
            metadata={
                "id": str(uuid.uuid4()),
                "location": {
                    "type": "Point",
                    "coordinates": [-82.35526376821747, 28.174933445526445]  # [longitude, latitude]
                }
            }
        ),
        Document(
            page_content=MEXICALI_BLISS,
            name="MEXICALI BLISS",
            metadata={
                "id": str(uuid.uuid4()),
                "location": {
                    "type": "Point",
                    "coordinates": [-82.33195054736757, 28.173903340587756]  # [longitude, latitude]
                }
            }
        ),
        Document(
            page_content=FIESTA_VERDE,
            name="FIESTA VERDE",
            metadata={
                "id": str(uuid.uuid4()),
                "location": {
                    "type": "Point",
                    "coordinates": [-82.34799031649871, 28.145102700392137]  # [longitude, latitude]
                }
            }
        ),
        Document(
            page_content=CANTINA_ROJA,
            name="CANTINA ROJA",
            metadata={
                "id": str(uuid.uuid4()),
                "location": {
                    "type": "Point",
                    "coordinates": [-82.3795401996739, 28.19073320615236]  # [longitude, latitude]
                }
            }
        ),
        Document(
            page_content=PUEBLA_PLAZA,
            name="PUEBLA PLAZA",
            metadata={
                "id": str(uuid.uuid4()),
                "location": {
                    "type": "Point",
                    "coordinates": [-82.2822284923503, 28.199285719832115]
                }
            }
        ),
    ]

