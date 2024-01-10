import openai
from dotenv import load_dotenv
# from app.chains.BasicChatChain import basicOpenAICompletion
import websocket
import json
import os
import threading
import pandas as pd
import requests

load_dotenv()
exit_flag = False
openai.api_key = os.environ.get('OPENAI_API_KEY')


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    END = '\033[0m'  # Resets the color to default


functions = [
    {
        "name": "reformat_to_csv",
        "description": "Use this function to save an evaluation into a csv file",
        "parameters": {
            "type": "object",
            "properties": {

                "grades": {
                    "type": "array",
                    "description": "An array of objects, each containing the grade and details of a model.",
                    "items": {  # This is the schema for the list items
                        "type": "object",
                        "properties": {
                            "model_id": {
                                "type": "string",
                                "description": "The leter that identifies the model (e.g. A, B, etc.)",
                            },
                            "model_name": {
                                "type": "string",
                                "description": "The name of the model. Examples of model names are: RAG_paragraph_k3 or RAG_megaChunk_k3",
                            },
                            "grade": {
                                "type": "integer",
                                "description": "The grate associated to each model",
                            },

                        }
                    }
                },
                "explanation": {
                    "type": "string",
                    "description": "The explanation of the grade",
                }
            }
        }
    }
]


document_extract = """RAPPORT MONDIAL SUR LE DIABÈTE
PREMIÈRE PARTIE: LA CHARGE DE MORBIDITÉ MONDIALE DU DIABÈTE

1.1 MORTALITÉ PAR  HYPERGLYCÉMIE, DIABÈTE INCLUS
En 2012, 1,5 million de décès dans  le monde ont été directement  imputables au diabète, huitième  cause principale de décès chez  les deux sexes et cinquième cause  principale de décès chez les femmes  en 2012 .

Une glycémie supérieure à la  normale, même inférieure à la valeur  seuil de diagnostic du diabète,  est une importante source de  mortalité et de morbidité. Le critère  de diagnostic du diabète est une  glycémie à jeun ≥7,0 mmol/l – valeur  de diagnostic retenue sur la base  de complications microvasculaires  comme la rétinopathie diabétique. Le  risque d’affection macrovasculaire,  tels l’infarctus du myocarde ou

l’accident vasculaire cérébral,  commence toutefois à augmenter  bien avant que soit atteinte cette  valeur seuil du diagnostic

. Aussi,  pour mieux comprendre tous les  effets de la glycémie sur la mortalité,  est-il nécessaire d’examiner la  mortalité liée à la glycémie en tant  que facteur de risque. La charge totale des décès dus  à l’hyperglycémie en 2012 a été  estimée à 3,7 millions. Ce chiffre

et chez les hommes est plus élevée  dans la classe d’âge de 60 à 69 ans.

43 % de tous les décès imputables  à l’hyperglycémie surviennent  prématurément, avant l’âge de 70  ans – nombre de décès estimé à 1,6  million dans le monde. À l’échelle  mondiale, l’hyperglycémie est  responsable de 7 % environ des  décès chez les hommes de 20 à 69  ans et de 8 % des décès chez les  femmes de 20 à 69 ans. La Figure  2 montre que le pourcentage des  décès prématurés imputables à  l’hyperglycémie est plus élevé  dans les pays à revenu faible ou  intermédiaire que dans les pays à  revenu élevé, et plus élevé chez les  hommes que chez les femmes.

Les taux de mortalité par  hyperglycémie normalisés selon  l’âge, qui tiennent compte des  différences dans la structure de la  population, présentent des écarts
inclut 1,5 million de décès par  diabète et 2,2 millions de décès  supplémentaires dus aux maladies  cardiovasculaires, à l’insuffisance  rénale chronique et à la tuberculose  associée à une glycémie supérieure  à la normale. Son importance  montre que l’hyperglycémie est  responsable d’une lourde charge  de mortalité au-delà des décès  directement imputables au diabète.  Les décès par hyperglycémie sont  plus nombreux dans les pays à  revenu intermédiaire (de la tranche  supérieure) (1,5 million) et moins  nombreux dans les pays à faible  revenu (0,3 million).
Après l’âge de 50 ans, les pays à  revenu intermédiaire ont la plus  forte proportion de décès imputés  à l’hyperglycémie chez les hommes  et chez les femmes (voir la Figure 1).  Sauf dans les pays à revenu élevé,  la proportion de décès imputables  à l’hyperglycémie chez les femmes
sensibles selon les Régions de  l’OMS (Tableau 1). Les taux sont  plus élevés dans les Régions OMS  de la Méditerranée orientale, de  l’Asie du Sud-Est et de l’Afrique, les  taux étant sensiblement inférieurs  dans les autres Régions. Dans  les Régions OMS de l’Europe, de  l’Asie du Sud-Est et des Amériques, 111.3 110.9 111.1 72.6 63.9 82.8 139.6 140.2 138.3 55.7 46.5 64.5 115.3 101.8 129.1 67 65.8 67.8
les  taux  de  mortalité  par  hyperglycémie sont sensiblement  plus élevés chez les hommes que  chez les femmes.
Pendant la période de 2000 à 2012,  la proportion de décès prématurés  (chez les personnes de 20 à 69  ans) imputables à l’hyperglycémie
a augmenté chez les deux sexes  dans toutes les Régions OMS,  sauf chez les femmes de la Région  européenne de l’OMS (Figure 3). La  hausse de la proportion des décès  imputables à l’hyperglycémie était  supérieure dans la Région OMS du  Pacifique occidental, où le nombre  total des décès imputables à  l’hyperglycémie pendant cette  période a également augmenté,  passant de 490 000 à 944 000.
1.2 PRÉVALENCE DU DIABÈTE   ET DES fACTEURS   DE RISQUE ASSOCIÉS
L’OMS estime que 422 millions  d’adultes de plus de 18 ans vivaient  avec le diabète dans le monde en  2014 (de plus amples détails sur  la méthodologie sont donnés à  l’annexe B et dans la référence  bibliographique

. Selon les  estimations, les personnes vivant  avec le diabète étaient plus  nombreuses dans les Régions OMS

de l’Asie du Sud-Est et du Pacifique  occidental (voir le Tableau 2),  totalisant environ la moitié des cas  de diabète dans le monde.

Le nombre de personnes atteintes  de diabète (définies dans les  enquêtes comme les personnes  présentant un taux de glycémie  à jeun égal ou supérieur à 7,0  mmol/l ou sous médication contre  le diabète/l’hyperglycémie) a  régulièrement progressé ces  dernières décennies, sous l’effet de  l’accroissement démographique,  de l’augmentation de l’âge moyen  de la population, et de la hausse  de la prévalence du diabète dans  chaque classe d’âge. À l’échelle  mondiale, le nombre des personnes  atteintes de diabète a sensiblement  augmenté entre 1980 et 2014,  passant de 108 millions aux chiffres  actuels qui sont environ quatre fois  supérieurs (voir le Tableau 2). Selon  les estimations, cette augmentation  résulte pour 40 % de l’accroissement  démographique et du vieillissement, 3.1% 7.1% 4 25 5% 8.3% 18 62 5.9% 13.7% 6 43 5.3% 7.3% 33 64 4.1% 8.6% 17 96 4.4% 8.4% 29 131
pour 28 % d’une hausse de la  prévalence à des âges déterminés,  et pour 32 % de l’interaction des  deux facteurs . Ces 3 dernières décennies, la  prévalence
(normalisée pour  l’âge) du diabète a sensiblement  augmenté dans les pays à tous  les niveaux de revenu, reflétant la  hausse mondiale du nombre des  personnes en surpoids ou obèses.  La prévalence mondiale du diabète  a augmenté, passant de 4,7 %  en 1980 à 8,5 % en 2014, période  pendant laquelle la prévalence dans  chaque pays a augmenté ou est,  au mieux, restée inchangée
. Au  cours de cette dernière décennie, la  prévalence du diabète a progressé  plus rapidement dans les pays à  revenu faible ou intermédiaire que  dans les pays à revenu élevé (voir  la Figure 4a). La Région OMS de la
Méditerranée orientale a enregistré  la hausse la plus forte et elle est  désormais la Région de l’OMS où  la prévalence est la plus élevée  (13,7 %) (voir la Figure 4b).
La distinction entre le diabète de  type 1 et le diabète de type 2 n’est  pas toujours aisée en raison des  examens de laboratoire relativement  complexes requis pour évaluer la  fonction pancréatique. Aussi n’existe- t-il pas d’estimations mondiales  distinctes de la prévalence du  diabète de type 1 et de la prévalence  du diabète de type 2.
Notre connaissance de l’incidence  du diabète de type 1 concerne pour  une grande part les enfants et elle  repose sur des initiatives concertées  destinées à établir des registres  normalisés des nouveaux cas dans  le monde, fondés sur la population,  comme le projet DIAMOND de l’OMS
. À l’échelle mondiale,  ces registres ont relevé des écarts  importants dans l’incidence et la  prévalence du diabète de type 1, de  plus de 60 à moins de 0,5 cas annuels  pour 100 000 enfants de moins de 15  ans ; les variations pouvant être dues  aux écarts liés à la vérification des cas.  Parmi les sites étudiés dans le projet  DIAMOND de l’OMS, le diabète  de type 1 est surtout répandu dans  les populations scandinaves ainsi  qu’en Sardaigne et au Koweït, et  beaucoup moins courant en Asie et  en Amérique latine
. Les données  concernant l’Afrique subsaharienne  et de grandes parties de l’Amérique  latine font généralement défaut.  Depuis ces quelques dernières  décennies, l’incidence annuelle  semble augmenter régulièrement de  3 % environ dans les pays à revenu  élevé .
Précédemment observé surtout  chez les personnes âgées ou d’âge  moyen, le diabète de type 2 touche  de plus en plus fréquemment les  enfants et les jeunes. Le diabète de  type 2 est rarement diagnostiqué  et, vu la complexité des études  destinées à évaluer le nombre de  nouveaux cas, il n’existe quasiment  pas de données sur l’incidence  réelle. Dans les pays à revenu élevé,  la prévalence du diabète de type  2 est fréquemment plus élevée  chez les personnes démunies
.  Il existe peu de données sur le  gradient social du diabète dans les  pays à revenu faible ou intermédiaire,  mais selon les données existantes,  bien que la prévalence du diabète  soit souvent plus élevée chez les  personnes nanties, cette tendance s’inverse dans certains pays à revenu  intermédiaire .
La proportion des cas de diabète  de type 2 non diagnostiqués varie  sensiblement – selon une étude  récente des données provenant  de sept pays, entre 24 et 62 % des  personnes atteintes de diabète  n’étaient pas diagnostiquées et ne  suivaient pas de traitement
.  L’analyse des données issues des  enquêtes STEPS de l’OMS réalisées  dans 11 pays fait apparaître des  écarts importants dans la proportion  des personnes non diagnostiquées  et non traitées : parmi les personnes  dont la glycémie mesurée était égale  ou supérieure à la valeur seuil de  diagnostic du diabète, entre 6 et 70 %  avaient été diagnostiquées positives  pour le diabète et entre 4 et 66 %  prenaient des médicaments pour  réduire leur glycémie . Même dans
les pays à revenu élevé, la proportion  des cas de diabète non diagnostiqués  peut atteindre de 30 à 50 % .
La fréquence des cas précédemment  non diagnostiqués de diabète  pendant la grossesse et de  diabète gestationnel varie entre  les populations, mais elle touche  probablement de 10 à 25 %  des grossesses
. Selon des  estimations, la plupart (de 75 à 90 %)  des cas d’hyperglycémie pendant la  grossesse sont des cas de diabète  gestationnel .
Une activité physique régulière  réduit le risque de diabète et  l’hyperglycémie, et elle est un facteur  important d’équilibre énergétique  général, de maîtrise du poids et de  prévention de l’obésité – tous les  risques liés à la prévalence future  du diabète
. La cible mondiale  d’une réduction relative de 10 % de  la sédentarité est par conséquent  fortement associée à la cible  mondiale de la suppression de ce  risque lié au diabète.
La prévalence de la sédentarité à  l’échelle mondiale est cependant une  source d’inquiétude croissante. En  2010, dernière année pour laquelle  des données sont disponibles, le  quart ou presque des adultes de  plus de 18 ans ne pratiquaient pas  l’activité physique hebdomadaire  minimum recommandée et étaient  classés comme ayant une activité  physique insuffisante
. Dans  toutes les Régions de l’OMS et dans  tous les groupes de pays (selon le  revenu), les femmes étaient moins  actives que les hommes, 27 % des  femmes et 20 % des hommes étant  classés comme ayant une activité  physique insuffisante. La sédentarité
est d’une ampleur inquiétante chez  les adolescents, 84 % des filles et  78 % des garçons ne suivant pas  les recommandations minimales  relatives à l’activité physique  pour cet âge. La prévalence de la  sédentarité est plus élevée dans  les pays à revenu élevé où elle est  près de deux fois supérieure à celle  des pays à faible revenu. Parmi les  Régions de l’OMS, la Région de la  Méditerranée orientale avait la plus  forte prévalence de sédentarité chez  les adultes et chez les adolescents.
Le surpoids et l’obésité sont  étroitement liés au diabète.  Malgré la cible volontaire mondiale  d’interrompre l’avancée de l’obésité  d’ici à 2025
, le surpoids et  l’obésité ont progressé dans presque  tous les pays. En 2014, dernière  année pour laquelle des estimations  mondiales sont disponibles, plus d’un  adulte de plus de 18 ans sur trois  était en surpoids et plus d’un sur 10  était obèse. Les femmes en surpoids  ou obèses étaient plus nombreuses  que les hommes. La prévalence de  l’obésité la plus élevée a été relevée  dans la Région OMS des Amériques  et la plus faible dans la Région OMS  de l’Asie du Sud-Est (voir la Figure  5a). La proportion de personnes en  surpoids ou obèses augmente avec  le niveau de revenu du pays. Les pays  à revenu élevé ou intermédiaire ont  une prévalence du surpoids et de  l’obésité qui est plus de deux fois  celle des pays à faible revenu (voir la  Figure 5b).
1.3 CHARGE DE MORBIDITÉ   ET ÉVOLUTION DES  COMPLICATIONS DU DIABÈTE
Le diabète, s’il est mal maîtrisé, peut  être cause de cécité, d’insuffisance  rénale, d’amputation des membres  inférieurs et de plusieurs autres
conséquences à long terme qui  entravent sérieusement la qualité  de vie. Il n’existe pas d’estimations  mondiales  des  insuffisances  rénales terminales, des maladies  cardiovasculaires, des amputations  des membres inférieurs ou des  complications de la grossesse  liées au diabète, bien que ces  affections touchent de nombreuses  personnes vivant avec le diabète.  Là où des données sont disponibles  – principalement fournies par des  pays à revenu élevé – la prévalence,  l’incidence et les tendances varient  sensiblement entre les pays .
La rétinopathie diabétique était  responsable de 1,9 % des troubles  visuels modérés ou graves dans le  monde et de 2,6 % des cas de cécité  en 2010
. Des études indiquent  que la prévalence de toutes les  rétinopathies chez les diabétiques  est de 35 % tandis que celle de  la rétinopathie proliférative (qui  menace la vision) est de 7 %
. Les  taux de rétinopathie sont cependant  plus élevés chez : les diabétiques de  type 1 ; les personnes atteintes de  diabète depuis plus longtemps ;  les populations caucasiennes (vs  asiatiques) ; voire chez les personnes  de  statut  socioéconomique  inférieur .
Les données totalisées de 54 pays  montrent qu’au moins 80 % des cas  d’insuffisance rénale terminale sont  dus au diabète, à l’hypertension  ou à un ensemble des deux  affections
. La proportion des  cas d’insuffisance rénale terminale  imputable au seul diabète se situe  entre 12 et 55 %. L’incidence de
l’insuffisance rénale terminale est  jusqu’à 10 fois plus élevée chez les  adultes atteints de diabète que  chez les adultes non diabétiques. La  prévalence de l’insuffisance rénale  terminale dépend fortement de  l’accès à la dialyse et au traitement  de suppléance rénale – très variable  selon les pays (dans certains cas au  sein d’un même pays).
Les adultes atteints de diabète ont  toujours eu un taux de maladies  cardiovasculaires de deux à trois  fois plus élevé que les adultes  non diabétiques
. Le risque  cardiovasculaire  augmente  parallèlement à la hausse des taux  de glycémie à jeun, même avant  que soient atteints les seuils de  diagnostic du diabète
. Les  quelques pays d’Amérique du nord,  la Scandinavie et le Royaume-Uni  de Grande Bretagne et d’Irlande du  Nord qui ont étudié les tendances  temporelles de l’incidence des  maladies  cardiovasculaires  (infarctus du myocarde, accident  vasculaire cérébral ou mortalité  par maladie cardiovasculaire) font  état d’importantes réductions,  ces 20 dernières années, chez  les diabétiques de type 1 et  de type 2
, toutefois moins  importantes que la réduction dans  la population non diabétique. Ce  recul a été imputé à la baisse de  la prévalence du tabagisme et à  l’amélioration de la prise en charge  du diabète et des facteurs de risque  cardiovasculaire associés. Le diabète semble accroître  sensiblement le risque d’amputation  des membres inférieurs lié à des ulcérations des pieds qui  s’infectent et ne guérissent pas
. Les taux d’amputation dans  les populations chez lesquelles un  diabète a été diagnostiqué sont  généralement de 10 à 20 fois plus  élevés que dans les populations  non diabétiques et, cette dernière  décennie, ils ont oscillé entre 1,5  et 3,5 cas pour 1000 personnes  par an dans les populations chez  lesquelles un diabète avait été  diagnostiqué. Il est encourageant  de noter, selon plusieurs études,  une réduction de 40 à 60 % des  taux d’amputation chez les adultes  diabétiques au cours de ces 10 à  15 dernières années au Royaume- Uni, en Suède, au Danemark, en  Espagne, aux États-Unis d’Amérique  et en Australie
. Il n’existe pas de  données de ce type pour les pays à  revenu faible ou intermédiaire.
1.4 RÉSUMÉ
Le nombre de diabétiques dans le  monde a quadruplé depuis 1980.  L’accroissement démographique  et le vieillissement de la population  ont contribué à cette hausse, mais  n’en sont pas la seule cause. La  prévalence (normalisée selon l’âge)  du diabète progresse dans toutes  les régions. La prévalence mondiale  a doublé entre 1980 et 2014,  témoignant d’une augmentation  du surpoids et de l’obésité.  La prévalence augmente plus  rapidement dans les pays à revenu  faible ou intermédiaire.

La glycémie commence à influer sur  la morbidité et la mortalité même  avant que soit atteinte la valeur  seuil de diagnostic du diabète. Le  diabète et une glycémie supérieure

à la normale sont conjointement  responsables de 3,7 millions de  décès, dont un grand nombre  pourrait être évité.

Les chiffres et les tendances  présentés dans cette section  ont des incidences sur la santé  et le bien-être des populations,  et sur les systèmes de santé. Les  complications du diabète ont de  graves effets sur les personnes  qui en souffrent et leur impact se  fait également sentir au niveau  de la population. Le diabète  menace sérieusement la santé des  populations.
RÉfÉRENCES BIBLIOGRAPHIQUES
1. WHO Mortality Database [base de données en ligne, en anglais]. Genève, Organisation mondiale  de la Santé, (http://apps.who.int/ healthinfo/statistics/mortality/causeofdeath_query/, consulté le 12  janvier 2016).

2. Singh GM, Danaei G, Farzadfar F, Stevens GA, Woodward M, Wormser D et al. The age-specific quantitative  effects of metabolic risk factors on cardiovascular diseases and diabetes: a pooled analysis. PLoS One 2013 ;  8(7):e65174.

3. Danaei G, Lawes CM, Vander HS, Murray CJ, Ezzati M. Global and regional mortality from ischaemic heart  disease and stroke attributable to higher-than-optimum blood glucose concentration: comparative risk  assessment. Lancet. 2006 ;368:(9548)1651–1659.

4. NCD Risk Factor Collaboration (NCD-RisC). Worldwide trends in diabetes since 1980: a pooled analysis of  751 population-based studies with 4*4 million participants. Lancet 2016 ; published online April 7. http:// dx.doi.org/10.1016/S0140-6736(16)00618-8.

5. Incidence and trends of childhood type 1 diabetes worldwide, 1990 1999. Diabetes Medicine.  2006 ;23:(8)857–866.

6. Tuomilehto J. The emerging global epidemic of type 1 diabetes. Current Diabetes Reports. 2013 ;13:  (6)795–804.

7. Patterson CC, Dahlquist GG, Gyurus E, Green A, Soltesz G. EURODIAB Study Group Incidence trends for  childhood type 1 diabetes in Europe during 1989–2003 and predicted new cases 2005–20: a multicentre  prospective registration study. Lancet. 2009 ;373:2027–2033. 8. Dabelea D. The accelerating epidemic of childhood diabetes. Lancet. 2009 ;373:(9680)1999–2000.

9. Gale EAM. The rise of childhood type 1 diabetes in the 20th century. Diabetes. 2002 ;51:3353–3361.

10. Diabetes: equity and social determinants. In Equity, social determinants and public health programmes. Blas  E, Kuru A, eds. Genève, Organisation mondiale de la Santé, 2010.

11. Gakidou E., Mallinger L., Abbott-Klafter J., Guerrero R., Villalpando S., Ridaura RL., et al. Traitement du  diabète et des facteurs de risque cardiovasculaire associés dans sept pays : comparaison des données  d’enquêtes nationales de santé par examen. Bulletin de l’Organisation mondiale de la Santé, 2011 ;  89:(3)172–183.

12. Tracking universal health coverage: first global monitoring report. Genève, Organisation mondiale de la  Santé, 2015.

13. Beagley J, Guariguata L, Weil C, Motala AA. Global estimates of undiagnosed diabetes in adults. Diabetes  Res Clin Pract. 2014 ;103:150–160.

14. Jiwani A, Marseille E, Lohse N, Damm P, Hod M, Kahn JG. Gestational diabetes mellitus: results from  a survey of country prevalence and practices. Journal of Maternal-Fetal Neonatal Medicine. 2012 ;25:  (6)600–610.

15. Guariguata L, Linnenkamp U, Beagley J, Whiting DR, Cho NH. Global estimates of the prevalence of  hyperglycaemia in pregnancy. Diabetes Res Clin Pract. 2014 ;103, (2) 176–185.

16. Global status report on noncommunicable diseases 2014. Genève, Organisation mondiale de la Santé,  2014.

17. Plan d’action mondial pour la lutte contre les maladies non transmissibles 2013-2020. Genève, Organisation  mondiale de la Santé, 2013.

18. United States Renal Data System. International Comparisons. In United States Renal Data System. 2014  USRDS annual data report: Epidemiology of kidney disease in the United States. Bethesda (MD): National  Institutes of Health, National Institute of Diabetes and Digestive and Kidney Diseases ; 2014:188–210.

19. Moxey PW, Gogalniceanu P, Hinchliffe RJ, Loftus IM, Jones KJ, Thompson MM, et al. Lower extremity  amputations – a review of global variability in incidence. Diabetic Medicine. 2011 ;28:(10)1144–1153.

20. Bourne RR, Stevens GA, White RA, Smith JL, Flaxman SR, Price H, et al. Causes of vision loss worldwide,  1990–2010: a systematic analysis. Lancet Global Health. 2013 ;1:(6)e339-e349.

21. Yau JW, Rogers SL, Kawasaki R, Lamoureux EL, Kowalski JW, Bek T, et al. Global prevalence and major risk  factors of diabetic retinopathy. Diabetes Care. 2012 ;35:(3)556–564.

22. Emerging Risk Factors Collaboration. Sarwar N, Gao P, Seshasai SR, Gobin R, Kaptoge S, Di Angelantonio E.  Diabetes mellitus, fasting blood glucose concentration, and risk of vascular disease: a collaborative meta- analysis of 102 prospective studies. Lancet. 2010 Jun 26 ;375(9733):2215–22.

23. Barengo NC, Katoh S, Moltchanov V, Tajima N, Tuomilehto J. The diabetes-cardiovascular risk paradox:  results from a Finnish population-based prospective study. European Heart Journal. 2008 ;29:(15)1889 1895. L’immense majorité des cas de  diabète dans le monde sont de type  2
"""


def test_question(question, models_to_test):
    evaluation_input_message = """
Evaluate the answers of the {n} models below to the question: "{question}".
{responses}
"""
    responses_string = ""
    responses = []
    ws_event = threading.Event()  # Define event for synchronization

    def on_open(ws, model_data):
        ws.send(json.dumps(model_data))

    def on_message(ws, message, stop_thread_ref, model):
        nonlocal responses_string

        response_json = json.loads(message)
        if response_json.get('context'):
            print(f"{Colors.BLUE}{response_json['context']}{Colors.END}")

        ws_event.set()  # Set the event when message is received

        if response_json.get('data') or response_json.get('error'):
            # Set to True to terminate the WebSocket loop
            if response_json.get('data'):
                gpt_response = response_json['data'].get(
                    'choices')[0]['message']['content']
                print(gpt_response)
                responses_string += "\n\nModel " + \
                    model['model_id']+": " + \
                    model['model_name']+"\n" + gpt_response
                responses.append(
                    {
                        'model_name': model['model_name'],
                        'model_id': model['model_id'],
                        'model_response': gpt_response
                    })
            stop_thread_ref[0] = True
            ws.close()

    def on_error(ws, error, stop_thread_ref):
        print("WebSocket Error:", error)
        # Set to True to terminate the WebSocket loop
        stop_thread_ref[0] = True
        ws.close()  # Close the WebSocket connection on error
        ws_event.set()

    def on_close(ws, close_status_code, close_msg):
        print("WebSocket connection closed with code:", close_status_code)
        ws.close()  # Close the WebSocket connection on close

    def run_ws(model):
        stop_thread_ref = [False]
        ws = websocket.WebSocketApp(
            model['url'],
            on_open=lambda ws: on_open(ws, model['data']),
            on_message=lambda ws, message: on_message(
                ws, message, stop_thread_ref, model),
            on_error=lambda ws, error: on_error(ws, error, stop_thread_ref),
            on_close=on_close
        )
        while not stop_thread_ref[0]:
            ws.run_forever()

    def run_request(model):
        nonlocal responses_string

        response = requests.post(model['url'], json=model['data'])

        if response.status_code == 200:
            gpt_response = response.json().get('response', 'ERROR fetching response')
            print(gpt_response)
            responses_string += "\n\nModel " + \
                model['model_id']+": " + \
                model['model_name']+"\n" + str(gpt_response)
            responses.append(
                {
                    'model_name': model['model_name'],
                    'model_id': model['model_id'],
                    'model_response': gpt_response
                })
        else:
            print(
                f"HTTP request error for model: {model['model_name']}. Status code: {response.status_code}")

    threads = []  # Create a list to store all the threads

    # Loop through the models
    for model in models_to_test:
        # HANDLE WEBSOCKETS
        if model['type'] == 'ws':
            ws_event.clear()

            # Start the WebSocket connection in a separate thread
            ws_thread = threading.Thread(target=run_ws, args=(model,))
            ws_thread.start()
            threads.append(ws_thread)  # Append the thread to the list

            # Wait for the message or timeout after 10 seconds
            ws_event.wait(timeout=10)

            # If the thread is still running after the timeout, it's likely stuck
            if ws_thread.is_alive():
                print(f"Timeout for model: {model['model_name']}")
                # Give it a timeout to ensure you don't wait indefinitely
                ws_thread.join(timeout=2)

        # HANDLE HTTP REQUESTS
        elif model['type'] == 'request':
            # HTTP request
            request_thread = threading.Thread(
                target=run_request, args=(model,))
            request_thread.start()
            threads.append(request_thread)

    # Ensure all threads have completed
    for t in threads:
        t.join()

    evaluation_input_message = evaluation_input_message.format(
        n=len(responses),
        question=question,
        responses=responses_string
    )

    # Return the list of responses
    return responses, evaluation_input_message


def qualitative_evaluation(question, evaluation_input_message, document_extract):
    print("Qualitative evaluation started...")

    system_message_qual_evaluation = """
Your goal is to evaluate the questions sent by the user according to how well do they align with the source material. 

DETAILED INSTRUCTIONS:
- Evaluate each of the answers with a 1-10 scale.
- Mention both the name and id of the models along side with your grade.
- Then, explain your reasoning behind your grading.
- Format your answer as described below.

SOURCE MATERIAL:
{source}

EXPECTED ANSWER FORMAT:
'
Model A: model_name - Grade: X
Model B: model_name - Grade: X

Explanation
'
    """

    system_message_qual_evaluation = system_message_qual_evaluation.format(
        source=document_extract)

    input_message = [{'role': 'system', 'content': system_message_qual_evaluation}, {
        'role': 'user', 'content': evaluation_input_message}]

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',  # 'gpt-4',  # 'gpt-3.5-turbo-16k'
        messages=input_message,
        temperature=0,
    )

    evaluation_result = response['choices'][0]['message']['content']

    print(f"{Colors.YELLOW}{evaluation_result}{Colors.END}")

    return evaluation_result


def saving_eval_in_csv(question, responses, evaluation):
    print("Saving evaluation started...")
    system_message_saving_csv = "Use your functions to reformat the evaluation result into a csv file."

    input_message = [{'role': 'system', 'content': system_message_saving_csv}, {
        'role': 'user', 'content': evaluation}]

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=input_message,
        temperature=0,
        functions=functions,
        function_call={"name": "reformat_to_csv"},
    )

    json_response = json.loads(response["choices"][0]["message"].get("function_call")[
        "arguments"])

    # Add question to the json response
    json_response["question"] = question

    # Add GPT answers to the json response
    for grade in json_response['grades']:
        for response in responses:
            if grade['model_name'] == response['model_name']:
                grade['model_response'] = response['model_response']
                break

    print(f"{Colors.GREEN}{json_response}{Colors.END}")
    print(json_response['grades'][0]['model_response'])

    # Extracting information from the JSON
    model_names = [grade['model_name'] for grade in json_response['grades']]
    grades = [grade['grade'] for grade in json_response['grades']]
    model_responses = [grade['model_response']
                       for grade in json_response['grades']]

    # Check if the CSV exists to append data; otherwise, create a new one
    if os.path.exists('output_v3.csv'):
        df = pd.read_csv('output_v3.csv')
    else:
        columns = ['doc_id', 'extract_id', 'question_id'] + model_names + ['explanation',
                                                                           'question'] + [f'answer_{model_name}' for model_name in model_names]
        df = pd.DataFrame(columns=columns)

    # Append data to the dataframe
    new_data = {
        'doc_id': 1,
        'extract_id': 1,
        'question_id': 1,
        **{model_name: grade for model_name, grade in zip(model_names, grades)},
        'explanation': json_response['explanation'],
        'question': json_response['question'],
        **{f'answer_{model_name}': response for model_name, response in zip(model_names, model_responses)}
    }
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

    # Save the dataframe back to the CSV
    df.to_csv('output_v3.csv', index=False)


questions = [
    "Qu'est-ce que le diabète de type 2 et comment sa prévalence a-t-elle évolué depuis 1980 ?",
    "Combien de décès dans le monde ont été causés par le diabète en 2012 ?",
    "Quelle est la valeur seuil de diagnostic du diabète basée sur la glycémie à jeun ?",
    "Quel est le pourcentage des décès prématurés imputables à l'hyperglycémie ?",
    "Quelles sont les régions de l'OMS avec la plus forte prévalence du diabète ?",
]

# question = "Qu'est-ce que le diabète de type 2 et comment sa prévalence a-t-elle évolué depuis 1980 ?"


system_message_A = """
You are EDDIE, a digital companion for diabetes patients.
Your goal is to answer user questios based on the context below. 
If the answer to user's question is not in the context provided, excuse your self and say that you lack information to be able to answer.

**CONTEXT:
Your context to answer the user question is the following:
{context}
"""

system_message_SPR = """
You are EDDIE, a digital companion for diabetes patients.
Your goal is to answer user questios based on the context below. 
If the answer to user's question is not in the context provided, excuse your self and say that you lack information to be able to answer.

**CONTEXT:
Your context to answer the user question is the following:
{context}

If necessary to answer the user question you can unpack information from the following priming statements:
{customer_profile}
"""


for question in questions:
    models = [
        {
            'model_name': 'RAG_paragraph_k3',
            'model_id': 'A',
            'type': 'ws',
            'url': "wss://llm-app-starterkit.herokuapp.com/ws",
            'data':
            {
                'chatlog': [
                    {
                        'role': 'system',
                        'content': system_message_A
                    },
                    {
                        'role': 'user',
                        'content': question
                    }],
                'knowledge_base': 'sanofi-chucks-s1',
                'model': 'gpt-3.5-turbo-0613',
                'functions': functions,
                'function_call': 'none',
            }
        },
        {
            'model_name': 'RAG_SPR_k3',
            'model_id': 'B',
            'type': 'ws',
            'url': "wss://llm-app-starterkit.herokuapp.com/rag_&_SPR",
            'data':
            {
                'chatlog': [
                    {
                        'role': 'system',
                        'content': system_message_SPR
                    },
                    {
                        'role': 'user',
                        'content': question
                    }],
                'knowledge_base': 'sanofi-chucks-s1',
                'model': 'gpt-3.5-turbo-0613',
                'functions': functions,
                'function_call': 'none',
            }
        },
        {
            'model_name': 'current_megachunkAPI_k3',
            'model_id': 'D',
            'type': 'request',
            'url': "https://t2dfunctionapp.azurewebsites.net/api/T2DFunctionApp?x-functions-key=7vlL3w6EEaV3W-mHdY2OpOGowlMLGfnPn7oxQ6zA5fsFAzFuQKXfbg==&Content-Type=application/json",
            'data':
            {
                'userid': 1,
                'conversationid': 1,
                'querystring': question,
            }
        }]

    """
    {
            'model_name': 'RAG_megachunk_k3',
            'model_id': 'C',
            'type': 'ws',
            'url': "wss://llm-app-starterkit.herokuapp.com/ws",
            'data':
            {
                'chatlog': [
                    {
                        'role': 'system',
                        'content': system_message_A
                    },
                    {
                        'role': 'user',
                        'content': question
                    }],
                'knowledge_base': 'sanofi-megachucks-s1',
                'model': 'gpt-3.5-turbo-0613',
                'functions': functions,
                'function_call': 'none',
            }
        },"""

    responses, evaluation_input_message = test_question(question, models)

    print("FINAL ANSWERS")
    print(responses)
    print(evaluation_input_message)

    evaluation_result = qualitative_evaluation(question,
                                               evaluation_input_message, document_extract)

    saving_eval_in_csv(question, responses, evaluation_result)
