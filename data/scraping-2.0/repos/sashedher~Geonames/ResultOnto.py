# import GeoOnto
from rdflib import Graph
import geocoder
import re
from pprint import pprint
# import ResultOntoIndiv
import openAI

print("\n\n-----------------------------India----------------------\n\n")
Feature_code={
                'A': ["country","state", "region"],
                'H': ["stream","lake"],
                'L': ["parks","area"],
                'P': ["city","village"],
                'R': ["road","railroad"],
                'S': ["spot","building","farm"],
                'T': ["mountain","hill","rock"],
                'U': ["undersea"],
                'V': ["forest","eath"]
            }
                        
SubFeature_Code={
          "A" : {
                  "ADM1"  :	"first-order administrative division.A primary administrative division of a country, such as a state in the United States",
                  "ADM1H" :	"historicalk first-order administrative division.A former first-order administrative division",
                  "ADM2"  :	"second-order administrative division.A subdivision of a first-order administrative division",
                  "ADM2H" :	"historical second-order administrative division.A former second-order administrative division",
                  "ADM3"  :	"third-order administrative division.A subdivision of a second-order administrative division",
                  "ADM3H" :	"historical third-order administrative division.A former third-order administrative division",
                  "ADM4"  :	"fourth-order administrative division.A subdivision of a third-order administrative division",
                  "ADM4H" :	"historical fourth-order administrative division.A former fourth-order administrative division",
                  "ADM5"  :	"fifth-order administrative division.A subdivision of a fourth-order administrative division",
                  "ADM5H" :	"historical fifth-order administrative division.A former fifth-order administrative division",
                  "ADMD"  :	"administrative division.An administrative division of a country, undifferentiated as to administrative level",
                  "ADMDH" :	"historical administrative division.A former administrative division of a political entity, undifferentiated as to administrative level",
                  "LTER"  :	"leased area.A tract of land leased to another country, usually for military installations",
                  "PCL"   :	"political entity	",
                  "PCLD"  :	"dependent political entity	",
                  "PCLF"  :	"freely associated state	",
                  "PCLH"  :	"historical political entity.A former political entity",
                  "PCLI"  :	"independent political entity	",
                  "PCLIX" :	"section of independent political entity	",
                  "PCLS"  :	"semi-independent political entity	",
                  "PRSH"  :	"parish.An ecclesiastical district",
                  "TERR"  :	"territory	",
                  "ZN"    :	"zone	",
                  "ZNB"   :	"buffer zone.A zone recognized as a buffer between two nations in which military presence is minimal or absent",

                },
          "H" : {
                  "AIRS"  : "seaplane landing area.A place on a waterbody where floatplanes land and take off",
                  "ANCH"  : "anchorage.An area where vessels may anchor",
                  "BAY" 	: "bay.A coastal indentation between two capes or headlands, larger than a cove but smaller than a gulf",
                  "BAYS"  : "bays.coastal indentations between two capes or headlands, larger than a cove but smaller than a gulf",
                  "BGHT"  : "bight(s).An open body of water forming a slight recession in a coastline",
                  "BNK" 	: "bank(s) .An elevation, typically located on a shelf, over which the depth of water is relatively shallow but sufficient for most surface navigation",
                  "BNKR"  : "stream bank .A sloping margin of a stream channel which normally confines the stream to its channel on land",
                  "BNKX"  : "section of bank.",
                  "BOG" 	: "bog(s).A wetland characterized by peat forming sphagnum moss, sedge, and other acid-water plants",
                  "CAPG"  : "icecap.A dome-shaped mass of glacial ice covering an area of mountain summits or other high lands; smaller than an ice sheet",
                  "CHN" 	: "channel.the deepest part of a stream, bay, lagoon, or strait, through which the main current flows",
                  "CHNL"  : "lake channel(s) .that part of a lake having water deep enough for navigation between islands, shoals, etc.",
                  "CHNM"  : "marine channel.that part of a body of water deep enough for navigation through an area otherwise not suitable",
                  "CHNN"  : "navigation channel.A buoyed channel of sufficient depth for the safe navigation of vessels",
                  "CNFL"  : "confluence.A place where two or more streams or intermittent streams flow together",
                  "CNL" 	: "canal.An artificial watercourse",
                  "CNLA"  : "aqueduct.A conduit used to carry water",
                  "CNLB"  : "canal bend.A conspicuously curved or bent section of a canal",
                  "CNLD"  : "drainage canal.An artificial waterway carrying water away from a wetland or from drainage ditches",
                  "CNLI"  : "irrigation canal.A canal which serves as a main conduit for irrigation water",
                  "CNLN"  : "navigation canal(s).A watercourse constructed for navigation of vessels",
                  "CNLQ"  : "abandoned canal.",
                  "CNLSB" : "underground irrigation canal(s).A gently inclined underground tunnel bringing water for irrigation from aquifers",
                  "CNLX"  : "section of canal.",
                  "COVE"  : "cove(s).A small coastal indentation, smaller than a bay",
                  "CRKT"  : "tidal creek(s).A meandering channel in a coastal wetland subject to bi-directional tidal currents",
                  "CRNT"  : "current.A horizontal flow of water in a given direction with uniform velocity",
                  "CUTF"  : "cutoff.A channel formed as a result of a stream cutting through a meander neck",
                  "DCK" 	: "dock(s).A waterway between two piers, or cut into the land for the berthing of ships",
                  "DCKB"  : "docking basin.A part of a harbor where ships dock",
                  "DOMG"  : "icecap dome.A comparatively elevated area on an icecap",
                  "DPRG"  : "icecap depression.A comparatively depressed area on an icecap",
                  "DTCH"  : "ditch.A small artificial watercourse dug for draining or irrigating the land",
                  "DTCHD" : "drainage ditch.A ditch which serves to drain the land",
                  "DTCHI" : "irrigation ditch.A ditch which serves to distribute irrigation water",
                  "DTCHM" : "ditch mouth(s).An area where a drainage ditch enters a lagoon, lake or bay",
                  "ESTY"  : "estuary.A funnel-shaped stream mouth or embayment where fresh water mixes with sea water under tidal influences",
                  "FISH"  : "fishing area.A fishing ground, bank or area where fishermen go to catch fish",
                  "FJD" 	: "fjord.A long, narrow, steep-walled, deep-water arm of the sea at high latitudes, usually along mountainous coasts",
                  "FJDS"  : "fjords.long, narrow, steep-walled, deep-water arms of the sea at high latitudes, usually along mountainous coasts",
                  "FLLS"  : "waterfall(s).A perpendicular or very steep descent of the water of a stream",
                  "FLLSX" : "section of waterfall(s).",
                  "FLTM"  : "mud flat(s).A relatively level area of mud either between high and low tide lines, or subject to flooding",
                  "FLTT"  : "tidal flat(s).A large flat area of mud or sand attached to the shore and alternately covered and uncovered by the tide",
                  "GLCR"  : "glacier(s).A mass of ice, usually at high latitudes or high elevations, with sufficient thickness to flow away from the source area in lobes, tongues, or masses",
                  "GULF"  : "gulf.A large recess in the coastline, larger than a bay",
                  "GYSR"  : "geyser.A type of hot spring with intermittent eruptions of jets of hot water and steam",
                  "HBR" 	: "harbor(s).A haven or space of deep water so sheltered by the adjacent land as to afford a safe anchorage for ships",
                  "HBRX"  : "section of harbor.",
                  "INLT"  : "inlet.A narrow waterway extending into the land, or connecting a bay or lagoon with a larger body of water",
                  "INLTQ" : "former inlet.An inlet which has been filled in, or blocked by deposits",
                  "LBED"  : "lake bed(s).A dried up or drained area of a former lake",
                  "LGN" 	: "lagoon.A shallow coastal waterbody, completely or partly separated from a larger body of water by a barrier island, coral reef or other depositional feature",
                  "LGNS"  : "lagoons.shallow coastal waterbodies, completely or partly separated from a larger body of water by a barrier island, coral reef or other depositional feature",
                  "LGNX"  : "section of lagoon.",
                  "LK"  	: "lake.A large inland body of standing water",
                  "LKC" 	: "crater lake.A lake in a crater or caldera",
                  "LKI" 	: "intermittent lake.",
                  "LKN" 	: "salt lake.An inland body of salt water with no outlet",
                  "LKNI"  : "intermittent salt lake.",
                  "LKO" 	: "oxbow lake.A crescent-shaped lake commonly found adjacent to meandering streams",
                  "LKOI"  : "intermittent oxbow lake.",
                  "LKS" 	: "lakes.large inland bodies of standing water",
                  "LKSB"  : "underground lake.A standing body of water in a cave",
                  "LKSC"  : "crater lakes.lakes in a crater or caldera",
                  "LKSI"  : "intermittent lakes.",
                  "LKSN"  : "salt lakes.inland bodies of salt water with no outlet",
                  "LKSNI" : "intermittent salt lakes.",
                  "LKX" 	: "section of lake.",
                  "MFGN"  : "salt evaporation ponds.diked salt ponds used in the production of solar evaporated salt",
                  "MGV" 	: "mangrove swamp.A tropical tidal mud flat characterized by mangrove vegetation",
                  "MOOR"  : "moor(s).An area of open ground overlaid with wet peaty soils",
                  "MRSH"  : "marsh(es).A wetland dominated by grass-like vegetation",
                  "MRSHN" : "salt marsh.A flat area, subject to periodic salt water inundation, dominated by grassy salt-tolerant plants",
                  "NRWS"  : "narrows.A navigable narrow part of a bay, strait, river, etc.",
                  "OCN" 	: "ocean.one of the major divisions of the vast expanse of salt water covering part of the earth",
                  "OVF" 	: "overfalls.An area of breaking waves caused by the meeting of currents or by waves moving against the current",
                  "PND" 	: "pond.A small standing waterbody",
                  "PNDI"  : "intermittent pond.",
                  "PNDN"  : "salt pond.A small standing body of salt water often in a marsh or swamp, usually along a seacoast",
                  "PNDNI" : "intermittent salt pond(s).",
                  "PNDS"  : "ponds.small standing waterbodies",
                  "PNDSF" : "fishponds.ponds or enclosures in which fish are kept or raised",
                  "PNDSI" : "intermittent ponds.",
                  "PNDSN" : "salt ponds.small standing bodies of salt water often in a marsh or swamp, usually along a seacoast",
                  "POOL"  : "pool(s).A small and comparatively still, deep part of a larger body of water such as a stream or harbor; or a small body of standing water",
                  "POOLI" : "intermittent pool.",
                  "RCH" 	: "reach.A straight section of a navigable stream or channel between two bends",
                  "RDGG"  : "icecap ridge.A linear elevation on an icecap",
                  "RDST"  : "roadstead.An open anchorage affording less protection than a harbor",
                  "RF"  	: "reef(s).A surface-navigation hazard composed of consolidated material",
                  "RFC" 	: "coral reef(s).A surface-navigation hazard composed of coral",
                  "RFX" 	: "section of reef.",
                  "RPDS"  : "rapids.A turbulent section of a stream associated with a steep, irregular stream bed",
                  "RSV" 	: "reservoir(s).An artificial pond or lake",
                  "RSVI"  : "intermittent reservoir.",
                  "RSVT"  : "water tank.A contained pool or tank of water at, below, or above ground level",
                  "RVN" 	: "ravine(s).A small, narrow, deep, steep-sided stream channel, smaller than a gorge",
                  "SBKH"  : "sabkha(s).A salt flat or salt encrusted plain subject to periodic inundation from flooding or high tides",
                  "SD"  	: "sound.A long arm of the sea forming a channel between the mainland and an island or islands; or connecting two larger bodies of water",
                  "SEA" 	: "sea.A large body of salt water more or less confined by continuous land or chains of islands forming a subdivision of an ocean",
                  "SHOL"  : "shoal(s).A surface-navigation hazard composed of unconsolidated material",
                  "SILL"  : "sill.the low part of an underwater gap or saddle separating basins, including a similar feature at the mouth of a fjord",
                  "SPNG"  : "spring(s).A place where ground water flows naturally out of the ground",
                  "SPNS"  : "sulphur spring(s).A place where sulphur ground water flows naturally out of the ground",
                  "SPNT"  : "hot spring(s).A place where hot ground water flows naturally out of the ground",
                  "STM" 	: "stream.A body of running water moving to a lower level in a channel on land",
                  "STMA"  : "anabranch.A diverging branch flowing out of a main stream and rejoining it downstream",
                  "STMB"  : "stream bend.A conspicuously curved or bent segment of a stream",
                  "STMC"  : "canalized stream.A stream that has been substantially ditched, diked, or straightened",
                  "STMD"  : "distributary(-ies).A branch which flows away from the main stream, as in a delta or irrigation canal",
                  "STMH"  : "headwaters.the source and upper part of a stream, including the upper drainage basin",
                  "STMI"  : "intermittent stream.",
                  "STMIX" : "section of intermittent stream.",
                  "STMM"  : "stream mouth(s).A place where a stream discharges into a lagoon, lake, or the sea",
                  "STMQ"  : "abandoned watercourse.A former stream or distributary no longer carrying flowing water, but still evident due to lakes, wetland, topographic or vegetation patterns",
                  "STMS"  : "streams.bodies of running water moving to a lower level in a channel on land",
                  "STMSB" : "lost river.A surface stream that disappears into an underground channel, or dries up in an arid area",
                  "STMX"  : "section of stream.",
                  "STRT"  : "strait.A relatively narrow waterway, usually narrower and less extensive than a sound, connecting two larger bodies of water",
                  "SWMP"  : "swamp.A wetland dominated by tree vegetation",
                  "SYSI"  : "irrigation system.A network of ditches and one or more of the following elements: water supply, reservoir, canal, pump, well, drain, etc.",
                  "TNLC"  : "canal tunnel.A tunnel through which a canal passes",
                  "WAD" 	: "wadi.A valley or ravine, bounded by relatively steep banks, which in the rainy season becomes a watercourse; found primarily in North Africa and the Middle East",
                  "WADB"  : "wadi bend.A conspicuously curved or bent segment of a wadi",
                  "WADJ"  : "wadi junction.A place where two or more wadies join",
                  "WADM"  : "wadi mouth.the lower terminus of a wadi where it widens into an adjoining floodplain, depression, or waterbody",
                  "WADS"  : "wadies.valleys or ravines, bounded by relatively steep banks, which in the rainy season become watercourses; found primarily in North Africa and the Middle East",
                  "WADX"  : "section of wadi.",
                  "WHRL"  : "whirlpool.A turbulent, rotating movement of water in a stream",
                  "WLL" 	: "well.A cylindrical hole, pit, or tunnel drilled or dug down to a depth from which water, oil, or gas can be pumped or brought to the surface",
                  "WLLQ"  : "abandoned well.",
                  "WLLS"  : "wells.cylindrical holes, pits, or tunnels drilled or dug down to a depth from which water, oil, or gas can be pumped or brought to the surface",
                  "WTLD"  : "wetland.An area subject to inundation, usually characterized by bog, marsh, or swamp vegetation",
                  "WTLDI" : "intermittent wetland.",
                  "WTRC"  : "watercourse.A natural, well-defined channel produced by flowing water, or an artificial channel designed to carry flowing water",
                  "WTRH"  : "waterhole(s).A natural hole, hollow, or small depression that contains water, used by man and animals, especially in arid areas"
            },
          "L" : {
                  "AGRC"  :  "agricultural colony.A tract of land set aside for agricultural settlement",
                  "AMUS"  :  "amusement park.Amusement Park are theme parks, adventure parks offering entertainment, similar to funfairs but with a fix location",
                  "AREA"  :  "area.A tract of land without homogeneous character or boundaries",
                  "BSND"  :  "drainage basin.An area drained by a stream",
                  "BSNP"  :  "petroleum basin.An area underlain by an oil-rich structural basin",
                  "BTL"   :  "battlefield.A site of a land battle of historical importance",
                  "CLG"   :  "clearing.An area in a forest with trees removed",
                  "CMN"   :  "common.A park or pasture for community use",
                  "CNS"   :  "concession area.A lease of land by a government for economic development, e.g., mining, forestry",
                  "COLF"  :  "coalfield.A region in which coal deposits of possible economic value occur",
                  "CONT"  :  "continent.continent: Europe, Africa, Asia, North America, South America, Oceania, Antarctica",
                  "CST"   :  "coast.A zone of variable width straddling the shoreline",
                  "CTRB"  :  "business center.A place where a number of businesses are located",
                  "DEVH"  :  "housing development.A tract of land on which many houses of similar design are built according to a development plan",
                  "FLD"   :  "field(s).An open as opposed to wooded area",
                  "FLDI"  :  "irrigated field(s).A tract of level or terraced land which is irrigated",
                  "GASF"  :  "gasfield.An area containing a subterranean store of natural gas of economic value",
                  "GRAZ"  :  "grazing area.An area of grasses and shrubs used for grazing",
                  "GVL"   :  "gravel area.An area covered with gravel",
                  "INDS"  :  "industrial area.An area characterized by industrial activity",
                  "LAND"  :  "arctic land.A tract of land in the Arctic",
                  "LCTY"  :  "locality.A minor area or place of unspecified or mixed character and indefinite boundaries",
                  "MILB"  :  "military base.A place used by an army or other armed service for storing arms and supplies, and for accommodating and training troops, a base from which operations can be initiated",
                  "MNA"   :  "mining area.An area of mine sites where minerals and ores are extracted",
                  "MVA"   :  "maneuver area.A tract of land where military field exercises are carried out",
                  "NVB"   :  "naval base.An area used to store supplies, provide barracks for troops and naval personnel, a port for naval vessels, and from which operations are initiated",
                  "OAS"   :  "oasis(-es).An area in a desert made productive by the availability of water",
                  "OILF"  :  "oilfield.An area containing a subterranean store of petroleum of economic value",
                  "PEAT"  :  "peat cutting area.An area where peat is harvested",
                  "PRK"   :  "park.An area, often of forested land, maintained as a place of beauty, or for recreation",
                  "PRT"   :  "port.A place provided with terminal and transfer facilities for loading and discharging waterborne cargo or passengers, usually located in a harbor",
                  "QCKS"  :  "quicksand.An area where loose sand with water moving through it may become unstable when heavy objects are placed at the surface, causing them to sink",
                  "RES"   :  "reserve.A tract of public land reserved for future use or restricted as to use",
                  "RESA"  :  "agricultural reserve.A tract of land reserved for agricultural reclamation and/or development",
                  "RESF"  :  "forest reserve.A forested area set aside for preservation or controlled use",
                  "RESH"  :  "hunting reserve.A tract of land used primarily for hunting",
                  "RESN"  :  "nature reserve.An area reserved for the maintenance of a natural habitat",
                  "RESP"  :  "palm tree reserve.An area of palm trees where use is controlled",
                  "RESV"  :  "reservation.A tract of land set aside for aboriginal, tribal, or native populations",
                  "RESW"  :  "wildlife reserve.A tract of public land reserved for the preservation of wildlife",
                  "RGN"   :  "region.An area distinguished by one or more observable physical or cultural characteristics",
                  "RGNE"  :  "economic region.A region of a country established for economic development or for statistical purposes",
                  "RGNH"  :  "historical region.A former historic area distinguished by one or more observable physical or cultural characteristics",
                  "RGNL"  :  "lake region.A tract of land distinguished by numerous lakes",
                  "RNGA"  :  "artillery range.A tract of land used for artillery firing practice",
                  "SALT"  :  "salt area.A shallow basin or flat where salt accumulates after periodic inundation",
                  "SNOW"  :  "snowfield.An area of permanent snow and ice forming the accumulation area of a glacier",
                  "TRB"   :  "tribal area.A tract of land used by nomadic or other tribes"
                },
          "P" : {
                    "PPL"   : "populated place.A city, town, village, or other agglomeration of buildings where people live and work",
                    "PPLA"  : "seat of a first-order administrative division.seat of a first-order administrative division (PPLC takes precedence over PPLA)",
                    "PPLA2" : "seat of a second-order administrative division",
                    "PPLA3" : "seat of a third-order administrative division",
                    "PPLA4" : "seat of a fourth-order administrative division",
                    "PPLA5" : "seat of a fifth-order administrative division",
                    "PPLC"  : "capital of a political entity",
                    "PPLCH" : "historical capital of a political entity.A former capital of a political entity",
                    "PPLF"  : "farm village.A populated place where the population is largely engaged in agricultural activities",
                    "PPLG"  : "seat of government of a political entity",
                    "PPLH"  : "historical populated place. a populated place that no longer exists",
                    "PPLL"  : "populated locality.An area similar to a locality but with a small group of dwellings or other buildings",
                    "PPLQ"  : "abandoned populated place",
                    "PPLR"  : "religious populated place.A populated place whose population is largely engaged in religious occupations",
                    "PPLS"  : "populated places.cities, towns, villages, or other agglomerations of buildings where people live and work",
                    "PPLW"  : "destroyed populated place.A village, town or city destroyed by a natural disaster, or by war",
                    "PPLX"  : "section of populated place",
                    "STLMT" : "israeli settlement"
              },
          "R" : {
                "CSWY"  :   "causeway.A raised roadway across wet ground or shallow water",
                "OILP"  :   "oil pipeline.A pipeline used for transporting oil",
                "PRMN"  :   "promenade.A place for public walking, usually along a beach front",
                "PTGE"  :   "portage.A place where boats, goods, etc., are carried overland between navigable waters",
                "RD"    :   "road.An open way with improved surface for transportation of animals, people and vehicles",
                "RDA"   :   "ancient road    the remains of a road used by ancient cultures",
                "RDB"   :   "road bend.A conspicuously curved or bent section of a road",
                "RDCUT" :   "road cut.An excavation cut through a hill or ridge for a road",
                "RDJCT" :   "road .A place where two or more roads join",
                "RJCT"  :   "railroad junction.A place where two or more railroad tracks join",
                "RR"    :   "railroad.A permanent twin steel-rail track on which freight and passenger cars move long distances",
                "RRQ"   :   "abandoned railroad",
                "RTE"   :   "caravan route.The route taken by caravans",
                "RYD"   :   "railroad yard.A system of tracks used for the making up of trains, and switching and storing freight cars",
                "ST"    :   "street.A paved urban thoroughfare",
                "STKR"  :   "stock route.A route taken by livestock herds",
                "TNL"   :   "tunnel.A subterranean passageway for transportation",
                "TNLN"  :   "natural tunnel.A cave that is open at both ends",
                "TNLRD" :   "road tunnel.A tunnel through which a road passes",
                "TNLRR" :   "railroad tunnel.A tunnel through which a railroad passes",
                "TNLS"  :   "tunnels     subterranean passageways for transportation",
                "TRL"   :   "trail.A path, track, or route used by pedestrians, animals, or off-road vehicles"
                },
          "S" : {
                "ADMF"  : "administrative facility.A government building",
                "AGRF"  : "agricultural facility.A building and/or tract of land used for improving agriculture",
                "AIRB"  : "airbase.An area used to store supplies, provide barracks for air force personnel, hangars and runways for aircraft, and from which operations are initiated",
                "AIRF"  : "airfield.A place on land where aircraft land and take off; no facilities provided for the commercial handling of passengers and cargo",
                "AIRH"  : "heliport.A place where helicopters land and take off",
                "AIRP"  : "airport.A place where aircraft regularly land and take off, with runways, navigational aids, and major facilities for the commercial handling of passengers and cargo",
                "AIRQ"  : "abandoned airfield   ",
                "AIRT"  : "terminal.Airport facilities for the handling of freight and passengers",
                "AMTH"  : "amphitheater.An oval or circular structure with rising tiers of seats about a stage or open space",
                "ANS"   : "archaeological/prehistoric site.A place where archeological remains, old structures, or cultural artifacts are located",
                "AQC"   : "aquaculture facility facility or area for the cultivation of aquatic animals and plants, especially fish, shellfish, and seaweed, in natural or controlled marine or freshwater environments; underwater agriculture",
                "ARCH"  : "arch.A natural or man-made structure in the form of an arch",
                "ARCHV" : "archive.A place or institution where documents are preserved",
                "ART"   : "piece of art.A piece of art, like a sculpture, painting. In contrast to monument (MNMT) it is not commemorative.",
                "ASTR"  : "astronomical station.A point on the earth whose position has been determined by observations of celestial bodies",
                "ASYL"  : "asylum.A facility where the insane are cared for and protected",
                "ATHF"  : "athletic field.A tract of land used for playing team sports, and athletic track and field events",
                "ATM"   : "automatic teller machine.An unattended electronic machine in a public place, connected to a data system and related equipment and activated by a bank customer to obtain cash withdrawals and other banking services.",
                "BANK"  : "bank.A business establishment in which money is kept for saving or commercial purposes or is invested, supplied for loans, or exchanged.",
                "BCN"   : "beacon.A fixed artificial navigation mark",
                "BDG"   : "bridge.A structure erected across an obstacle such as a stream, road, etc., in order to carry roads, railroads, and pedestrians across",
                "BDGQ"  : "ruined bridge.A destroyed or decayed bridge which is no longer functional",
                "BLDA"  : "apartment building.A building containing several individual apartments",
                "BLDG"  : "building(s).A structure built for permanent use, as a house, factory, etc.",
                "BLDO"  : "office building  commercial building where business and/or services are conducted",
                "BP"    : "boundary marker.A fixture marking a point along a boundary",
                "BRKS"  : "barracks.A building for lodging military personnel",
                "BRKW"  : "breakwater.A structure erected to break the force of waves at the entrance to a harbor or port",
                "BSTN"  : "baling station.A facility for baling agricultural products",
                "BTYD"  : "boatyard.A waterside facility for servicing, repairing, and building small vessels",
                "BUR"   : "burial cave(s).A cave used for human burials",
                "BUSTN" : "bus station.A facility comprising ticket office, platforms, etc. for loading and unloading passengers",
                "BUSTP" : "bus stop.A place lacking station facilities",
                "CARN"  : "cairn.A heap of stones erected as a landmark or for other purposes",
                "CAVE"  : "cave(s).An underground passageway or chamber, or cavity on the side of a cliff",
                "CH"    : "church.A building for public Christian worship",
                "CMP"   : "camp(s).A site occupied by tents, huts, or other shelters for temporary use",
                "CMPL"  : "logging camp.A camp used by loggers",
                "CMPLA" : "labor camp.A camp used by migrant or temporary laborers",
                "CMPMN" : "mining camp.A camp used by miners",
                "CMPO"  : "oil camp.A camp used by oilfield workers",
                "CMPQ"  : "abandoned camp   ",
                "CMPRF" : "refugee camp.A camp used by refugees",
                "CMTY"  : "cemetery.A burial place or ground",
                "COMC"  : "communication center.A facility, including buildings, antennae, towers and electronic equipment for receiving and transmitting information",
                "CRRL"  : "corral(s).A pen or enclosure for confining or capturing animals",
                "CSNO"  : "casino.A building used for entertainment, especially gambling",
                "CSTL"  : "castle.A large fortified building or set of buildings",
                "CSTM"  : "customs house.A building in a port where customs and duties are paid, and where vessels are entered and cleared",
                "CTHSE" : "courthouse.A building in which courts of law are held",
                "CTRA"  : "atomic center.A facility where atomic research is carried out",
                "CTRCM" : "community center.A facility for community recreation and other activities",
                "CTRF"  : "facility center.A place where more than one facility is situated",
                "CTRM"  : "medical center.A complex of health care buildings including two or more of the following: hospital, medical school, clinic, pharmacy, doctor's offices, etc.",
                "CTRR"  : "religious center.A facility where more than one religious activity is carried out, e.g., retreat, school, monastery, worship",
                "CTRS"  : "space center.A facility for launching, tracking, or controlling satellites and space vehicles",
                "CVNT"  : "convent.A building where a community of nuns lives in seclusion",
                "DAM"   : "dam.A barrier constructed across a stream to impound water",
                "DAMQ"  : "ruined dam.A destroyed or decayed dam which is no longer functional",
                "DAMSB" : "sub-surface dam.A dam put down to bedrock in a sand river",
                "DARY"  : "dairy.A facility for the processing, sale and distribution of milk or milk products",
                "DCKD"  : "dry dock.A dock providing support for a vessel, and means for removing the water so that the bottom of the vessel can be exposed",
                "DCKY"  : "dockyard.A facility for servicing, building, or repairing ships",
                "DIKE"  : "dike.An earth or stone embankment usually constructed for flood or stream control",
                "DIP"   : "diplomatic facility  office, residence, or facility of a foreign government, which may include an embassy, consulate, chancery, office of charge d'affaires, or other diplomatic, economic, military, or cultural mission",
                "DPOF"  : "fuel depot.An area where fuel is stored",
                "EST"   : "estate(s).A large commercialized agricultural landholding with associated buildings and other facilities",
                "ESTO"  : "oil palm plantation.An estate specializing in the cultivation of oil palm trees",
                "ESTR"  : "rubber plantation.An estate which specializes in growing and tapping rubber trees",
                "ESTSG" : "sugar plantation.An estate that specializes in growing sugar cane",
                "ESTT"  : "tea plantation.An estate which specializes in growing tea bushes",
                "ESTX"  : "section of estate    ",
                "FCL"   : "facility.A building or buildings housing a center, institute, foundation, hospital, prison, mission, courthouse, etc.",
                "FNDY"  : "foundry.A building or works where metal casting is carried out",
                "FRM"   : "farm.A tract of land with associated buildings devoted to agriculture",
                "FRMQ"  : "abandoned farm   ",
                "FRMS"  : "farms    tracts of land with associated buildings devoted to agriculture",
                "FRMT"  : "farmstead    the buildings and adjacent service areas of a farm",
                "FT" : "fort.A defensive structure or earthworks",
                "FY" : "ferry.A boat or other floating conveyance and terminal facilities regularly used to transport people and vehicles across a waterbody",
                "FYT"   : "ferry terminal.A place where ferries pick-up and discharge passengers, vehicles and or cargo",
                "GATE"  : "gate.A controlled access entrance or exit",
                "GDN"   : "garden(s).An enclosure for displaying selected plant or animal life",
                "GHAT"  : "ghat.A set of steps leading to a river, which are of religious significance, and at their base is usually a platform for bathing",
                "GHSE"  : "guest house.A house used to provide lodging for paying guests",
                "GOSP"  : "gas-oil separator plant.A facility for separating gas from oil",
                "GOVL"  : "local government office.A facility housing local governmental offices, usually a city, town, or village hall",
                "GRVE"  : "grave.A burial site",
                "HERM"  : "hermitage.A secluded residence, usually for religious sects",
                "HLT"   : "halting place.A place where caravans stop for rest",
                "HMSD"  : "homestead.A residence, owner's or manager's, on a sheep or cattle station, woolshed, outcamp, or Aboriginal outstation, specific to Australia and New Zealand",
                "HSE"   : "house(s).A building used as a human habitation",
                "HSEC"  : "country house.A large house, mansion, or chateau, on a large estate",
                "HSP"   : "hospital.A building in which sick or injured, especially those confined to bed, are medically treated",
                "HSPC"  : "clinic.A medical facility associated with a hospital for outpatients",
                "HSPD"  : "dispensary.A building where medical or dental aid is dispensed",
                "HSPL"  : "leprosarium.An asylum or hospital for lepers",
                "HSTS"  : "historical site.A place of historical importance",
                "HTL"   : "hotel.A building providing lodging and/or meals for the public",
                "HUT"   : "hut.A small primitive house",
                "HUTS"  : "huts small primitive houses",
                "INSM"  : "military installation.A facility for use of and control by armed forces",
                "ITTR"  : "research institute.A facility where research is carried out",
                "JTY"   : "jetty.A structure built out into the water at a river mouth or harbor entrance to regulate currents and silting",
                "LDNG"  : "landing.A place where boats receive or discharge passengers and freight, but lacking most port facilities",
                "LEPC"  : "leper colony.A settled area inhabited by lepers in relative isolation",
                "LIBR"  : "library.A place in which information resources such as books are kept for reading, reference, or lending.",
                "LNDF"  : "landfill.A place for trash and garbage disposal in which the waste is buried between layers of earth to build up low-lying land",
                "LOCK"  : "lock(s).A basin in a waterway with gates at each end by means of which vessels are passed from one water level to another",
                "LTHSE" : "lighthouse.A distinctive structure exhibiting a major navigation light",
                "MALL"  : "mall.A large, often enclosed shopping complex containing various stores, businesses, and restaurants usually accessible by common passageways.",
                "MAR"   : "marina.A harbor facility for small boats, yachts, etc.",
                "MFG"   : "factory  one or more buildings where goods are manufactured, processed or fabricated",
                "MFGB"  : "brewery  one or more buildings where beer is brewed",
                "MFGC"  : "cannery.A building where food items are canned",
                "MFGCU" : "copper works.A facility for processing copper ore",
                "MFGLM" : "limekiln.A furnace in which limestone is reduced to lime",
                "MFGM"  : "munitions plant.A factory where ammunition is made",
                "MFGPH" : "phosphate works.A facility for producing fertilizer",
                "MFGQ"  : "abandoned factory    ",
                "MFGSG" : "sugar refinery.A facility for converting raw sugar into refined sugar",
                "MKT"   : "market.A place where goods are bought and sold at regular intervals",
                "ML" : "mill(s).A building housing machines for transforming, shaping, finishing, grinding, or extracting products",
                "MLM"   : "ore treatment plant.A facility for improving the metal content of ore by concentration",
                "MLO"   : "olive oil mill.A mill where oil is extracted from olives",
                "MLSG"  : "sugar mill.A facility where sugar cane is processed into raw sugar",
                "MLSGQ" : "former sugar mill.A sugar mill no longer used as a sugar mill",
                "MLSW"  : "sawmill.A mill where logs or lumber are sawn to specified shapes and sizes",
                "MLWND" : "windmill.A mill or water pump powered by wind",
                "MLWTR" : "water mill.A mill powered by running water",
                "MN" : "mine(s).A site where mineral ores are extracted from the ground by excavating surface pits and subterranean passages",
                "MNAU"  : "gold mine(s).A mine where gold ore, or alluvial gold is extracted",
                "MNC"   : "coal mine(s).A mine where coal is extracted",
                "MNCR"  : "chrome mine(s).A mine where chrome ore is extracted",
                "MNCU"  : "copper mine(s).A mine where copper ore is extracted",
                "MNFE"  : "iron mine(s).A mine where iron ore is extracted",
                "MNMT"  : "monument.A commemorative structure or statue",
                "MNN"   : "salt mine(s).A mine from which salt is extracted",
                "MNQ"   : "abandoned mine",
                "MNQR"  : "quarry(-ies).A surface mine where building stone or gravel and sand, etc. are extracted",
                "MOLE"  : "mole.A massive structure of masonry or large stones serving as a pier or breakwater",
                "MSQE"  : "mosque.A building for public Islamic worship",
                "MSSN"  : "mission.A place characterized by dwellings, school, church, hospital and other facilities operated by a religious group for the purpose of providing charitable services and to propagate religion",
                "MSSNQ" : "abandoned mission    ",
                "MSTY"  : "monastery.A building and grounds where a community of monks lives in seclusion",
                "MTRO"  : "metro station    metro station (Underground, Tube, or Metro)",
                "MUS"   : "museum.A building where objects of permanent interest in one or more of the arts and sciences are preserved and exhibited",
                "NOV"   : "novitiate.A religious house or school where novices are trained",
                "NSY"   : "nursery(-ies).A place where plants are propagated for transplanting or grafting",
                "OBPT"  : "observation point.A wildlife or scenic observation point",
                "OBS"   : "observatory.A facility equipped for observation of atmospheric or space phenomena",
                "OBSR"  : "radio observatory.A facility equipped with an array of antennae for receiving radio waves from space",
                "OILJ"  : "oil pipeline junction.A section of an oil pipeline where two or more pipes join together",
                "OILQ"  : "abandoned oil well   ",
                "OILR"  : "oil refinery.A facility for converting crude oil into refined petroleum products",
                "OILT"  : "tank farm.A tract of land occupied by large, cylindrical, metal tanks in which oil or liquid petrochemicals are stored",
                "OILW"  : "oil well.A well from which oil may be pumped",
                "OPRA"  : "opera house.A theater designed chiefly for the performance of operas.",
                "PAL"   : "palace.A large stately house, often a royal or presidential residence",
                "PGDA"  : "pagoda.A tower-like storied structure, usually a Buddhist shrine",
                "PIER"  : "pier.A structure built out into navigable water on piles providing berthing for ships and recreation",
                "PKLT"  : "parking lot.An area used for parking vehicles",
                "PMPO"  : "oil pumping station.A facility for pumping oil through a pipeline",
                "PMPW"  : "water pumping station.A facility for pumping water from a major well or through a pipeline",
                "PO"    : "post office.A public building in which mail is received, sorted and distributed",
                "PP"    : "police post.A building in which police are stationed",
                "PPQ"   : "abandoned police post",
                "PRKGT" : "park gate.A controlled access to a park",
                "PRKHQ" : "park headquarters.A park administrative facility",
                "PRN"   : "prison.A facility for confining prisoners",
                "PRNJ"  : "reformatory.A facility for confining, training, and reforming young law offenders",
                "PRNQ"  : "abandoned prison ",
                "PS"    : "power station.A facility for generating electric power",
                "PSH"   : "hydroelectric power station.A building where electricity is generated from water power",
                "PSN"   : "nuclear power station    nuclear power station",
                "PSTB"  : "border post.A post or station at an international boundary for the regulation of movement of people and goods",
                "PSTC"  : "customs post.A building at an international boundary where customs and duties are paid on goods",
                "PSTP"  : "patrol post.A post from which patrols are sent out",
                "PYR"   : "pyramid.An ancient massive structure of square ground plan with four triangular faces meeting at a point and used for enclosing tombs",
                "PYRS"  : "pyramids.Ancient massive structures of square ground plan with four triangular faces meeting at a point and used for enclosing tombs",
                "QUAY"  : "quay.A structure of solid construction along a shore or bank which provides berthing for ships and which generally provides cargo handling facilities",
                "RDCR"  : "traffic circle.A road junction formed around a central circle about which traffic moves in one direction only",
                "RDIN"  : "intersection.A junction of two or more highways by a system of separate levels that permit traffic to pass from one to another without the crossing of traffic streams",
                "RECG"  : "golf course.A recreation field where golf is played",
                "RECR"  : "racetrack.A track where races are held",
                "REST"  : "restaurant.A place where meals are served to the public",
                "RET"   : "store.A building where goods and/or services are offered for sale",
                "RHSE"  : "resthouse.A structure maintained for the rest and shelter of travelers",
                "RKRY"  : "rookery.A breeding place of a colony of birds or seals",
                "RLG"   : "religious site.An ancient site of significant religious importance",
                "RLGR"  : "retreat.A place of temporary seclusion, especially for religious groups",
                "RNCH"  : "ranch(es).A large farm specializing in extensive grazing of livestock",
                "RSD"   : "railroad siding.A short track parallel to and joining the main track",
                "RSGNL" : "railroad signal.A signal at the entrance of a particular section of track governing the movement of trains",
                "RSRT"  : "resort.A specialized facility for vacation, health, or participation sports activities",
                "RSTN"  : "railroad station.A facility comprising ticket office, platforms, etc. for loading and unloading train passengers and freight",
                "RSTNQ" : "abandoned railroad station   ",
                "RSTP"  : "railroad stop.A place lacking station facilities where trains stop to pick up and unload passengers and freight",
                "RSTPQ" : "abandoned railroad stop  ",
                "RUIN"  : "ruin(s).A destroyed or decayed structure which is no longer functional",
                "SCH"   : "school   building(s) where instruction in one or more branches of knowledge takes place",
                "SCHA"  : "agricultural school.A school with a curriculum focused on agriculture",
                "SCHC"  : "college  the grounds and buildings of an institution of higher learning",
                "SCHL"  : "language school  Language Schools & Institutions",
                "SCHM"  : "military school.A school at which military science forms the core of the curriculum",
                "SCHN"  : "maritime school.A school at which maritime sciences form the core of the curriculum",
                "SCHT"  : "technical school post-secondary school with a specifically technical or vocational curriculum",
                "SECP"  : "State Exam Prep Centre   state exam preparation centres",
                "SHPF"  : "sheepfold.A fence or wall enclosure for sheep and other small herd animals",
                "SHRN"  : "shrine.A structure or place memorializing a person or religious concept",
                "SHSE"  : "storehouse.A building for storing goods, especially provisions",
                "SLCE"  : "sluice.A conduit or passage for carrying off surplus water from a waterbody, usually regulated by means of a sluice gate",
                "SNTR"  : "sanatorium.A facility where victims of physical or mental disorders are treated",
                "SPA"   : "spa.A resort area usually developed around a medicinal spring",
                "SPLY"  : "spillway.A passage or outlet through which surplus water flows over, around or through a dam",
                "SQR"   : "square.A broad, open, public area near the center of a town or city",
                "STBL"  : "stable.A building for the shelter and feeding of farm animals, especially horses",
                "STDM"  : "stadium.A structure with an enclosure for athletic games with tiers of seats for spectators",
                "STNB"  : "scientific research base.A scientific facility used as a base from which research is carried out or monitored",
                "STNC"  : "coast guard station.A facility from which the coast is guarded by armed vessels",
                "STNE"  : "experiment station.A facility for carrying out experiments",
                "STNF"  : "forest station.A collection of buildings and facilities for carrying out forest management",
                "STNI"  : "inspection station.A station at which vehicles, goods, and people are inspected",
                "STNM"  : "meteorological station.A station at which weather elements are recorded",
                "STNR"  : "radio station.A facility for producing and transmitting information by radio waves",
                "STNS"  : "satellite station.A facility for tracking and communicating with orbiting satellites",
                "STNW"  : "whaling station.A facility for butchering whales and processing train oil",
                "STPS"  : "steps    stones or slabs placed for ease in ascending or descending a steep slope",
                "SWT"   : "sewage treatment plant   facility for the processing of sewage and/or wastewater",
                "SYG"   : "synagogue.A place for Jewish worship and religious instruction",
                "THTR"  : "theater.A building, room, or outdoor structure for the presentation of plays, films, or other dramatic performances",
                "TMB"   : "tomb(s).A structure for interring bodies",
                "TMPL"  : "temple(s).An edifice dedicated to religious worship",
                "TNKD"  : "cattle dipping tank.A small artificial pond used for immersing cattle in chemically treated water for disease control",
                "TOLL"  : "toll gate/barrier    highway toll collection station",
                "TOWR"  : "tower.A high conspicuous structure, typically much higher than its diameter",
                "TRAM"  : "tram rail vehicle along urban streets (also known as streetcar or trolley)",
                "TRANT" : "transit terminal facilities for the handling of vehicular freight and passengers",
                "TRIG"  : "triangulation station.A point on the earth whose position has been determined by triangulation",
                "TRMO"  : "oil pipeline terminal.A tank farm or loading facility at the end of an oil pipeline",
                "TWO"   : "temp work office Temporary Work Offices",
                "UNIP"  : "university prep school   University Preparation Schools & Institutions",
                "UNIV"  : "university.An institution for higher learning with teaching and research facilities constituting a graduate school and professional schools that award master's degrees and doctorates and an undergraduate division that awards bachelor's degrees.",
                "USGE"  : "united states government establishment.A facility operated by the United States Government in Panama",
                "VETF"  : "veterinary facility.A building or camp at which veterinary services are available",
                "WALL"  : "wall.A thick masonry structure, usually enclosing a field or building, or forming the side of a structure",
                "WALLA" : "ancient wall the remains of a linear defensive stone structure",
                "WEIR"  : "weir(s).A small dam in a stream, designed to raise the water level or to divert stream flow through a desired channel",
                "WHRF"  : "wharf(-ves).A structure of open rather than solid construction along a shore or a bank which provides berthing for ships and cargo-handling facilities",
                "WRCK"  : "wreck    the site of the remains of a wrecked vessel",
                "WTRW"  : "waterworks.A facility for supplying potable water through a water source and a system of pumps and filtration beds",
                "ZNF"   : "free trade zone.An area, usually a section of a port, where goods may be received and shipped free of customs duty and of most customs regulations",
                "ZOO"   : "zoo.A zoological garden or park where wild animals are kept for exhibition"             
                },
          "T" : {
                "ASPH"  : "asphalt lake.A small basin containing naturally occurring asphalt",
                "ATOL"  : "atoll(s).A ring-shaped coral reef which has closely spaced islands on it encircling a lagoon",
                "BAR"   : "bar.A shallow ridge or mound of coarse unconsolidated material in a stream channel, at the mouth of a stream, estuary, or lagoon and in the wave-break zone along coasts",
                "BCH"   : "beach.A shore zone of coarse unconsolidated sediment that extends from the low-water line to the highest reach of storm waves",
                "BCHS"  : "beaches.A shore zone of coarse unconsolidated sediment that extends from the low-water line to the highest reach of storm waves",
                "BDLD"  : "badlands.An area characterized by a maze of very closely spaced, deep, narrow, steep-sided ravines, and sharp crests and pinnacles",
                "BLDR"  : "boulder field.A high altitude or high latitude bare, flat area covered with large angular rocks",
                "BLHL"  : "blowhole(s).A hole in coastal rock through which sea water is forced by a rising tide or waves and spurted through an outlet into the air",
                "BLOW"  : "blowout(s).A small depression in sandy terrain, caused by wind erosion",
                "BNCH"  : "bench.A long, narrow bedrock platform bounded by steeper slopes above and below, usually overlooking a waterbody",
                "BUTE"  : "butte(s).A small, isolated, usually flat-topped hill with steep sides",
                "CAPE"  : "cape.A land area, more prominent than a point, projecting into the sea and marking a notable change in coastal direction",
                "CFT"   : "cleft(s).A deep narrow slot, notch, or groove in a coastal cliff",
                "CLDA"  : "caldera.A depression measuring kilometers across formed by the collapse of a volcanic mountain",
                "CLF"   : "cliff(s).A high, steep to perpendicular slope overlooking a waterbody or lower area",
                "CNYN"  : "canyon.A deep, narrow valley with steep sides cutting into a plateau or mountainous area",
                "CONE"  : "cone(s).A conical landform composed of mud or volcanic material",
                "CRDR"  : "corridor.A strip or area of land having significance as an access way",
                "CRQ"   : "cirque.A bowl-like hollow partially surrounded by cliffs or steep slopes at the head of a glaciated valley",
                "CRQS"  : "cirques.bowl-like hollows partially surrounded by cliffs or steep slopes at the head of a glaciated valley",
                "CRTR"  : "crater(s).A generally circular saucer or bowl-shaped depression caused by volcanic or meteorite explosive action",
                "CUET"  : "cuesta(s).An asymmetric ridge formed on tilted strata",
                "DLTA"  : "delta.A flat plain formed by alluvial deposits at the mouth of a stream",
                "DPR"   : "depression(s).A low area surrounded by higher land and usually characterized by interior drainage",
                "DSRT"  : "desert.A large area with little or no vegetation due to extreme environmental conditions",
                "DUNE"  : "dune(s).A wave form, ridge or star shape feature composed of sand",
                "DVD"   : "divide.A line separating adjacent drainage basins",
                "ERG"   : "sandy desert.An extensive tract of shifting sand and sand dunes",
                "FAN"   : "fan(s).A fan-shaped wedge of coarse alluvium with apex merging with a mountain stream bed and the fan spreading out at a low angle slope onto an adjacent plain",
                "FORD"  : "ford.A shallow part of a stream which can be crossed on foot or by land vehicle",
                "FSR"   : "fissure.A crack associated with volcanism",
                "GAP"   : "gap.A low place in a ridge, not used for transportation",
                "GRGE"  : "gorge(s).A short, narrow, steep-sided section of a stream valley",
                "HDLD"  : "headland.A high projection of land extending into a large body of water beyond the line of the coast",
                "HLL"   : "hill.A rounded elevation of limited extent rising above the surrounding land with local relief of less than 300m",
                "HLLS"  : "hills.rounded elevations of limited extent rising above the surrounding land with local relief of less than 300m",
                "HMCK"  : "hammock(s).A patch of ground, distinct from and slightly above the surrounding plain or wetland. Often occurs in groups",
                "HMDA"  : "rock desert.A relatively sand-free, high bedrock plateau in a hot desert, with or without a gravel veneer",
                "INTF"  : "interfluve.A relatively undissected upland between adjacent stream valleys",
                "ISL"   : "island.A tract of land, smaller than a continent, surrounded by water at high water",
                "ISLET" : "islet.small island, bigger than rock, smaller than island.",
                "ISLF"  : "artificial island.An island created by landfill or diking and filling in a wetland, bay, or lagoon",
                "ISLM"  : "mangrove island.A mangrove swamp surrounded by a waterbody",
                "ISLS"  : "islands.tracts of land, smaller than a continent, surrounded by water at high water",
                "ISLT"  : "land-tied island.A coastal island connected to the mainland by barrier beaches, levees or dikes",
                "ISLX"  : "section of island.",
                "ISTH"  : "isthmus.A narrow strip of land connecting two larger land masses and bordered by water",
                "KRST"  : "karst area.A distinctive landscape developed on soluble rock such as limestone characterized by sinkholes, caves, disappearing streams, and underground drainage",
                "LAVA"  : "lava area.An area of solidified lava",
                "LEV"   : "levee.A natural low embankment bordering a distributary or meandering stream; often built up artificially to control floods",
                "MESA"  : "mesa(s).A flat-topped, isolated elevation with steep slopes on all sides, less extensive than a plateau",
                "MND"   : "mound(s).A low, isolated, rounded hill",
                "MRN"   : "moraine.A mound, ridge, or other accumulation of glacial till",
                "MT"    : "mountain.An elevation standing high above the surrounding area with small summit area, steep slopes and local relief of 300m or more",
                "MTS"   : "mountains.A mountain range or a group of mountains or high ridges",
                "NKM"   : "meander neck.A narrow strip of land between the two limbs of a meander loop at its narrowest point",
                "NTK"   : "nunatak.A rock or mountain peak protruding through glacial ice",
                "NTKS"  : "nunataks.rocks or mountain peaks protruding through glacial ice",
                "PAN"   : "pan.A near-level shallow, natural depression or basin, usually containing an intermittent lake, pond, or pool",
                "PANS"  : "pans.A near-level shallow, natural depression or basin, usually containing an intermittent lake, pond, or pool",
                "PASS"  : "pass.A break in a mountain range or other high obstruction, used for transportation from one side to the other [See also gap]",
                "PEN"   : "peninsula.An elongate area of land projecting into a body of water and nearly surrounded by water",
                "PENX"  : "section of peninsula.",
                "PK"    : "peak.A pointed elevation atop a mountain, ridge, or other hypsographic feature",
                "PKS"   : "peaks.pointed elevations atop a mountain, ridge, or other hypsographic features",
                "PLAT"  : "plateau.An elevated plain with steep slopes on one or more sides, and often with incised streams",
                "PLATX" : "section of plateau.",
                "PLDR"  : "polder.An area reclaimed from the sea by diking and draining",
                "PLN"   : "plain(s).An extensive area of comparatively level to gently undulating land, lacking surface irregularities, and usually adjacent to a higher area",
                "PLNX"  : "section of plain.",
                "PROM"  : "promontory(-ies).A bluff or prominent hill overlooking or projecting into a lowland",
                "PT"    : "point.A tapering piece of land projecting into a body of water, less prominent than a cape",
                "PTS"   : "points.tapering pieces of land projecting into a body of water, less prominent than a cape",
                "RDGB"  : "beach ridge.A ridge of sand just inland and parallel to the beach, usually in series",
                "RDGE"  : "ridge(s).A long narrow elevation with steep sides, and a more or less continuous crest",
                "REG"   : "stony desert.A desert plain characterized by a surface veneer of gravel and stones",
                "RK"    : "rock.A conspicuous, isolated rocky mass",
                "RKFL"  : "rockfall.An irregular mass of fallen rock at the base of a cliff or steep slope",
                "RKS"   : "rocks.conspicuous, isolated rocky masses",
                "SAND"  : "sand area.A tract of land covered with sand",
                "SBED"  : "dry stream bed.A channel formerly containing the water of a stream",
                "SCRP"  : "escarpment.A long line of cliffs or steep slopes separating level surfaces above and below",
                "SDL"   : "saddle.A broad, open pass crossing a ridge or between hills or mountains",
                "SHOR"  : "shore.A narrow zone bordering a waterbody which covers and uncovers at high and low water, respectively",
                "SINK"  : "sinkhole.A small crater-shape depression in a karst area",
                "SLID"  : "slide.A mound of earth material, at the base of a slope and the associated scoured area",
                "SLP"   : "slope(s).A surface with a relatively uniform slope angle",
                "SPIT"  : "spit.A narrow, straight or curved continuation of a beach into a waterbody",
                "SPUR"  : "spur(s).A subordinate ridge projecting outward from a hill, mountain or other elevation",
                "TAL"   : "talus slope.A steep concave slope formed by an accumulation of loose rock fragments at the base of a cliff or steep slope",
                "TRGD"  : "interdune trough(s).A long wind-swept trough between parallel longitudinal dunes",
                "TRR"   : "terrace.A long, narrow alluvial platform bounded by steeper slopes above and below, usually overlooking a waterbody",
                "UPLD"  : "upland.An extensive interior region of high land with low to moderate surface relief",
                "VAL"   : "valley.An elongated depression usually traversed by a stream",
                "VALG"  : "hanging valley.A valley the floor of which is notably higher than the valley or shore to which it leads; most common in areas that have been glaciated",
                "VALS"  : "valleys.elongated depressions usually traversed by a stream",
                "VALX"  : "section of valley.",
                "VLC"   : "volcano.A conical elevation composed of volcanic materials with a crater at the top"
                },
          "U" : {
                "APNU"  : "apron.A gentle slope, with a generally smooth surface, particularly found around groups of islands and seamounts",
                "ARCU"  : "arch.A low bulge around the southeastern end of the island of Hawaii",
                "ARRU"  : "arrugado.An area of subdued corrugations off Baja California",
                "BDLU"  : "borderland.A region adjacent to a continent, normally occupied by or bordering a shelf, that is highly irregular with depths well in excess of those typical of a shelf",
                "BKSU"  : "banks.elevations, typically located on a shelf, over which the depth of water is relatively shallow but sufficient for safe surface navigation",
                "BNKU"  : "bank.An elevation, typically located on a shelf, over which the depth of water is relatively shallow but sufficient for safe surface navigation",
                "BSNU"  : "basin.A depression more or less equidimensional in plan and of variable extent",
                "CDAU"  : "cordillera.An entire mountain system including the subordinate ranges, interior plateaus, and basins",
                "CNSU"  : "canyons.Relatively narrow, deep depressions with steep sides, the bottom of which generally has a continuous slope",
                "CNYU"  : "canyon.A relatively narrow, deep depression with steep sides, the bottom of which generally has a continuous slope",
                "CRSU"  : "continental rise.A gentle slope rising from oceanic depths towards the foot of a continental slope",
                "DEPU"  : "deep.A localized deep area within the confines of a larger feature, such as a trough, basin or trench",
                "EDGU"  : "shelf edge.A line along which there is a marked increase of slope at the outer margin of a continental shelf or island shelf",
                "ESCU"  : "escarpment (or scarp).An elongated and comparatively steep slope separating flat or gently sloping areas",
                "FANU"  : "fan.A relatively smooth feature normally sloping away from the lower termination of a canyon or canyon system",
                "FLTU"  : "flat.A small level or nearly level area",
                "FRZU"  : "fracture zone.An extensive linear zone of irregular topography of the sea floor, characterized by steep-sided or asymmetrical ridges, troughs, or escarpments",
                "FURU"  : "furrow.A closed, linear, narrow, shallow depression",
                "GAPU"  : "gap.A narrow break in a ridge or rise",
                "GLYU"  : "gully.A small valley-like feature",
                "HLLU"  : "hill.An elevation rising generally less than 500 meters",
                "HLSU"  : "hills.elevations rising generally less than 500 meters",
                "HOLU"  : "hole.A small depression of the sea floor",
                "KNLU"  : "knoll.An elevation rising generally more than 500 meters and less than 1,000 meters and of limited extent across the summit",
                "KNSU"  : "knolls.elevations rising generally more than 500 meters and less than 1,000 meters and of limited extent across the summits",
                "LDGU"  : "ledge.A rocky projection or outcrop, commonly linear and near shore",
                "LEVU"  : "levee.An embankment bordering a canyon, valley, or seachannel",
                "MESU"  : "mesa.An isolated, extensive, flat-topped elevation on the shelf, with relatively steep sides",
                "MNDU"  : "mound.A low, isolated, rounded hill",
                "MOTU"  : "moat.An annular depression that may not be continuous, located at the base of many seamounts, islands, and other isolated elevations",
                "MTU"   : "mountain.A well-delineated subdivision of a large and complex positive feature",
                "PKSU"  : "peaks.prominent elevations, part of a larger feature, either pointed or of very limited extent across the summit",
                "PKU"   : "peak.A prominent elevation, part of a larger feature, either pointed or of very limited extent across the summit",
                "PLNU"  : "plain.A flat, gently sloping or nearly level region",
                "PLTU"  : "plateau.A comparatively flat-topped feature of considerable extent, dropping off abruptly on one or more sides",
                "PNLU"  : "pinnacle.A high tower or spire-shaped pillar of rock or coral, alone or cresting a summit",
                "PRVU"  : "province.A region identifiable by a group of similar physiographic features whose characteristics are markedly in contrast with surrounding areas",
                "RDGU"  : "ridge.A long narrow elevation with steep sides",
                "RDSU"  : "ridges.long narrow elevations with steep sides",
                "RFSU"  : "reefs.surface-navigation hazards composed of consolidated material",
                "RFU"   : "reef.A surface-navigation hazard composed of consolidated material",
                "RISU"  : "rise.A broad elevation that rises gently, and generally smoothly, from the sea floor",
                "SCNU"  : "seachannel.A continuously sloping, elongated depression commonly found in fans or plains and customarily bordered by levees on one or two sides",
                "SCSU"  : "seachannels.continuously sloping, elongated depressions commonly found in fans or plains and customarily bordered by levees on one or two sides",
                "SDLU"  : "saddle.A low part, resembling in shape a saddle, in a ridge or between contiguous seamounts",
                "SHFU"  : "shelf.A zone adjacent to a continent (or around an island) that extends from the low water line to a depth at which there is usually a marked increase of slope towards oceanic depths",
                "SHLU"  : "shoal.A surface-navigation hazard composed of unconsolidated material",
                "SHSU"  : "shoals.Hazards to surface navigation composed of unconsolidated material",
                "SHVU"  : "shelf valley.A valley on the shelf, generally the shoreward extension of a canyon",
                "SILU"  : "sill.the low part of a gap or saddle separating basins",
                "SLPU"  : "slope.the slope seaward from the shelf edge to the beginning of a continental rise or the point where there is a general reduction in slope",
                "SMSU"  : "seamounts.elevations rising generally more than 1,000 meters and of limited extent across the summit",
                "SMU"   : "seamount.An elevation rising generally more than 1,000 meters and of limited extent across the summit",
                "SPRU"  : "spur.A subordinate elevation, ridge, or rise projecting outward from a larger feature",
                "TERU"  : "terrace.A relatively flat horizontal or gently inclined surface, sometimes long and narrow, which is bounded by a steeper ascending slope on one side and by a steep descending slope on the opposite side",
                "TMSU"  : "tablemounts (or guyots).seamounts having a comparatively smooth, flat top",
                "TMTU"  : "tablemount (or guyot).A seamount having a comparatively smooth, flat top",
                "TNGU"  : "tongue.An elongate (tongue-like) extension of a flat sea floor into an adjacent higher feature",
                "TRGU"  : "trough.A long depression of the sea floor characteristically flat bottomed and steep sided, and normally shallower than a trench",
                "TRNU"  : "trench.A long, narrow, characteristically very deep and asymmetrical depression of the sea floor, with relatively steep sides",
                "VALU"  : "valley.A relatively shallow, wide depression, the bottom of which usually has a continuous gradient",
                "VLSU"  : "valleys.A relatively shallow, wide depression, the bottom of which usually has a continuous gradient"            
                },
          "V" : {
                "BUSH"  : "bush(es).A small clump of conspicuous bushes in an otherwise bare area",
                "CULT"  : "cultivated area.An area under cultivation",
                "FRST"  : "forest(s).An area dominated by tree vegetation",
                "FRSTF" : "fossilized forest.A forest fossilized by geologic processes and now exposed at the earth's surface",
                "GROVE" : "grove.A small wooded area or collection of trees growing closely together, occurring naturally or deliberately planted",
                "GRSLD" : "grassland.An area dominated by grass vegetation",
                "GRVC"  : "coconut grove.A planting of coconut trees",
                "GRVO"  : "olive grove.A planting of olive trees",
                "GRVP"  : "palm grove.A planting of palm trees",
                "GRVPN" : "pine grove.A planting of pine trees",
                "HTH"   : "heath.An upland moor or sandy area dominated by low shrubby vegetation including heather",
                "MDW"   : "meadow.A small, poorly drained area dominated by grassy vegetation",
                "OCH"   : "orchard(s).A planting of fruit or nut trees",
                "SCRB"  : "scrubland.An area of low trees, bushes, and shrubs stunted by some environmental limitation",
                "TREE"  : "tree(s).A conspicuous tree used as a landmark",
                "TUND"  : "tundra.A marshy, treeless, high latitude plain, dominated by mosses, lichens, and low shrub vegetation under permafrost conditions",
                "VIN"   : "vineyard.A planting of grapevines",
                "VINS"  : "vineyards.plantings of grapevines",
                "ll"    : "not available"
                },
}

def geoid(ip):
    g = geocoder.geonames(ip, key='sashedher')
    ip_id = g.geonames_id
    # print(ip_id)
    return ip_id


def load_geo_onto():
    g_aBox = Graph()
    g_aBox.parse("Dataset.ttl")
    g_aBox.parse("ontology_v3.2.rdf")
    return g_aBox


def OntoMatch1(strg) -> bool:
    Onto = re.compile(r"http://www.geonames.org/ontology")
    return Onto.match(strg) is not None


def OntoMatch2(strg) -> bool:
    Onto = re.compile(r"http://www.w3.org/2003/01/geo/wgs84_pos")
    return Onto.match(strg) is not None


def about_info(qry, grph):
    alternateName = "http://www.geonames.org/ontology#alternateName"
    officialName = "http://www.geonames.org/ontology#officialName"
    # postalCode="http://www.geonames.org/ontology#postalCode"
    namespace = "http://www.geonames.org/ontology#"
    coords = "http://www.w3.org/2003/01/geo/wgs84_pos#"
    i = 0
    result_r = grph.query(qry)
    about = dict()
    for subj, pred, obj in result_r:
        if str(pred) != alternateName and str(pred) != officialName:
            if OntoMatch1(str(pred)):
                i = i + 1
                x = str(pred)
                x = x.replace(namespace, '')
                # print("{:>20} {:>30} ".format(x,obj))
                about[x] = str(obj)
            if OntoMatch2(str(pred)):
                x = str(pred)
                x = x.replace(coords, '')
                # print("{:>20} {:>30} ".format(x,obj))
                about[x] = str(obj)
    # print(about)
    return about


def cities_info(qry, grph):
    result_r = grph.query(qry)
    type(result_r)
    cities = dict()
    i = 0
    for s, o in result_r:
        i = i + 1
        # print(s, p, o)
        cities[str(s)] = str(o)

    return cities

def query_aboutinfo(ipid):
    qery = """
    
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        PREFIX gn: <http://www.geonames.org/ontology#>
        PREFIX cc: <http://creativecommons.org/ns#>
        PREFIX wgs84_pos: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX :<https://sws.geonames.org/>
        
        construct {<https://sws.geonames.org/""" + str(ipid) + """/> ?p ?o}
            WHERE {<https://sws.geonames.org/""" + str(ipid) + """/> ?p ?o}    
    """
    return qery


def query_citiesinfo(ipid, pred):
    qery = """

    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX gn: <http://www.geonames.org/ontology#>
    PREFIX cc: <http://creativecommons.org/ns#>
    PREFIX wgs84_pos: <http://www.w3.org/2003/01/geo/wgs84_pos#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX :<https://sws.geonames.org/>

    select ?name ?s
      WHERE {?s gn:"""+pred+""" <https://sws.geonames.org/"""+str(ipid)+"""/>.
            ?s gn:name ?name.}
      
    """
    return qery

def temp1(ctrycode,fcl=None)->str:
  if(fcl is  None):
    st="This geographical location belongs to the country "+ ctrycode.upper()+". "
    return st
  namespace="https://www.geonames.org/ontology#"
  fcl=fcl.replace(namespace,'')
  fc=Feature_code[fcl[0]]
  st="This geographical location belongs to the country "+ ctrycode.upper()+" and it is classified as "
  for ftr in fc:
    st=st+str(ftr.upper())+" "
  # print(st)
  # print('FeatureCode is       :{}'.format(Feature_code[z[0]]))
  st=st+". "
  st=st+"\nThe more specific details are, it is "+SubFeature_Code[fcl[0]][fcl[2:]].upper()
  return st

def temp2(offname,popln='000',poscd='000000')->str:
  if(poscd!='000000'):
    x="The official name of the location is "+ offname.upper()+" which has a population of around "+ popln .upper()+" with the postal code is "+poscd
  else:
    if(popln!='000'):
      x="The official name of the location is "+ offname.upper()+" which has a population of around "+ popln .upper()+". "
    else:
      x="The official name of the location is "+ offname.upper()
  return x;

def temp3(map,lat,logt)->str:
  x="we can locate this location help of latitude and longitude "+ lat +" & "+logt+" respectively or either with the help of location map: "+map
  return x;

def temp4(wiki)->str:
  x="More details about feature is explained in this wikipedia document : "+wiki
  return x;

def keyWordSentence(pred_list,about):
    res="The location name is "+about['name']+". "
    for x in pred_list:

      try:

        if(x == 'countryCode'):
          res += "This geographical location belongs to the country "+about[x]+". "

        if(x == 'featureCode'):
          namespace="https://www.geonames.org/ontology#"
          fcl= about[x]
          fcl=fcl.replace(namespace,'')
          print(fcl)
          fc=Feature_code[fcl[0]]
          st="This geographical location is classified as "
          for ftr in fc:
            st=st+str(ftr.upper())+" "
          st=st+". "
          st=st+"\nThe more specific details are, it is "+SubFeature_Code[fcl[0]][fcl[2:]].upper()
          print(st)
          res += st 


        if(x== 'postalCode' or x == 'population'):
          res += "The "+x+" for "+about['name']+" is "+about[x]+". "
        
        if(x=='Coordinates'):
          res += about['name']+" is cooordinated at a point lat is "+about['lat']+" and long is "+about['long']+". "

        if(x == 'locationMap' or x == 'wikipediaArticle'):
          res += "The "+x+" is avilable in "+about[x]+" "
      except:
        res= res
    print(res)
    return res

def load_loc_info(id,type):
    g_aBox = Graph()

    url='https://sws.geonames.org/'+str(id)+'/'+type+'.rdf'
    try:
        g_aBox.parse(url)
        # print("{}:  {}".format(i,url))
    except:
        print("This feature does not have this type of rdf file or invalid id")

    return g_aBox

def get_result(_id,pred_list):
    # g_abox = GeoOnto.load_geo_onto()
    print(pred_list)
    ip_id = geoid(_id)
    print("input id for str is :"+str(ip_id))
    # if(ip_id is None):

    g_abox = load_loc_info(ip_id,'about')
    qry = query_aboutinfo(ip_id)
    about = about_info(qry, g_abox)
    pprint(about)
    del g_abox

    g_abox = load_loc_info(ip_id,'nearby')
    qry = query_citiesinfo(ip_id, "nearby")
    nearbys = cities_info(qry, g_abox)
    del g_abox

    g_abox = load_loc_info(ip_id,'neighbours')
    qry = query_citiesinfo(ip_id, "neighbour")
    neighbours = cities_info(qry, g_abox)
    del g_abox

    g_abox = load_loc_info(ip_id,'contains')
    qry = query_citiesinfo(ip_id, "parentFeature")
    contains = cities_info(qry, g_abox)

    sentences = dict()
    try:
        sentences['temp1']=temp1(about['countryCode'],about['featureCode'])
    except:
        sentences['temp1']=temp1('IN')

    try:
        sentences['temp2']=temp2(about['name'],about['population'],about['postalCode']) # add postal only when available
    except:
      try:
        sentences['temp2']=temp2(about['name'],about['population'])
      except:
        sentences['temp2']=temp2(about['name'])
    sentences['temp3']=temp3(about['locationMap'],about['lat'],about['long'])
    try:
      sentences['temp4']=temp4(about['wikipediaArticle'])
    except:
      sentences['temp4']="No wikipedia article found"

    
    # pprint(about)

    # print("\n----------------------------- nearby cities------------------\n")
    # pprint(nearbys)

    # print("\n----------------------------- neighbour cities------------------\n")
    # pprint(neighbours)
    #
    # print("\n----------------------------- contains cities------------------\n")
    # pprint(contains)
    
    
    keysent=keyWordSentence(pred_list,about)
    sentences['openAI']= openAI.generate_sentence(keysent)
    result = {'about': about, 'nearbys': nearbys, 'neighbours': neighbours, 'contains': contains, 'sentences':sentences}
    
    nersen= ""
    if 'Nearbys' in pred_list:
      if(len(nearbys) == 0):
        nersen ="There is no info about nearby places for "+about['name']
      else:
        nersen = "The following list of locations are near to "+about['name']+" are "
      result['nersen'] = nersen;
    
    neisen= ""
    if 'Neighbours' in pred_list:
      if(len(neighbours) == 0):
        neisen ="There is no info about neighbouring location for "+about['name']
      else:
        neisen = "The following list of locations are neighbour to"+about['name']+" are "
      result['neisen'] = neisen;
    
    consen= ""
    if 'contains' in pred_list:
      if(len(contains) == 0):
        consen ="There is no info about child locations for "+about['name']
      else:
        consen = "The following list of locations are contained inside "+about['name']+" are "
      result['consen'] = consen;
    
    count = 0
    colList, temp = [], []
    print(contains.keys())
    for a in contains.keys():
      temp.append(a)
      count+=1
      if count==5:
        colList.append(temp.copy())
        temp.clear()
        count = 0
    if len(temp)>0:
        colList.append(temp.copy())
    count = 0
    nerList, n1 = [], []
    print(contains.keys())
    for a in nearbys.keys():
      n1.append(a)
      count+=1
      if count==5:
        nerList.append(n1.copy())
        n1.clear()
        count = 0
    if len(n1)>0:
        nerList.append(n1.copy())
    count = 0
    neiList, n2 = [], []
    print(contains.keys())
    for a in neighbours.keys():
      n2.append(a)
      count+=1
      if count==5:
        neiList.append(n2.copy())
        n2.clear()
        count = 0
    if len(n2)>0:
        colList.append(n2.copy())
    print("sentence from OpenAi"+sentences['openAI'])
    
    result['nerList'] = nerList
    # print(result['containslist'])
    
    result['neiList'] = neiList
    # print(result['containslist'])
    
    
    result['containslist'] = colList
    result['nearbysize'] = len(nearbys)
    result['neighboursize'] = len(neighbours)
    result['containsize'] = len(contains)
    print(result['nearbys'])
    
    
    del g_abox
    
    
    return result
