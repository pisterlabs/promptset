from openAI_experiment import InfoDataBase, Experiment
import matplotlib.pyplot as plt
from mod_analysis import data_plot
import numpy as np
import scipy


def dave_smith_db():
    name = "Dave Smith GPT3"

    system_message = ("Imitate a social study experiment subject. "
                      "You will be given some information about a purely hypothetical person "
                      "called Dave Smith. Then you will be asked to assess his virtue to the "
                      "best of your capability on the scale from 0 to 100. Your score should "
                      "roughly match the percentile of Dave Smith in the human population.")

    pro = [
        "Dave Smith has volunteered at a local homeless shelter a few times in the past.",
        "Dave Smith has shown some potential for leadership skills in minor projects at work.",
        "Dave Smith enjoys playing music as a hobby and occasionally performs at casual gatherings.",
        "Dave Smith maintains an acceptable academic performance, meeting the basic requirements of his studies.",
        "Dave Smith occasionally participates in community clean-up drives to contribute to a cleaner environment.",
        "Dave Smith has offered occasional advice to individuals when available.",
        "Dave Smith is willing to lend an ear and provide support to friends and family when possible.",
        "Dave Smith sometimes donates to charitable organizations, showing occasional support for causes he cares about.",
        "Dave Smith cooperates with his team at work, fulfilling his responsibilities adequately.",
        "Dave Smith has an interest in cooking and occasionally prepares meals for friends and family to enjoy."
    ]

    contra = [
        "Dave Smith has frequently been late to meetings, causing significant inconvenience to colleagues and disrupting the schedule.",
        "Dave Smith consistently interrupts others during conversations, often disregarding their thoughts and opinions.",
        "Dave Smith has accumulated numerous traffic citations for various offenses, including reckless driving and running red lights.",
        "Dave Smith frequently procrastinates, often resulting in rushed and subpar submissions of assignments or tasks.",
        "Dave Smith struggles to control anger in stressful situations, leading to frequent outbursts and confrontations.",
        "Dave Smith habitually exaggerates his achievements and experiences when sharing stories, often distorting the truth.",
        "Dave Smith has a history of borrowing items from friends and acquaintances without returning them or acknowledging the act.",
        "Dave Smith lives in a consistently disorganized and cluttered personal space, making it difficult to maintain order.",
        "Dave Smith frequently cancels plans on short notice, displaying a lack of consideration for others' time and commitments.",
        "Dave Smith regularly ignores or forgets to respond to text messages and emails, causing communication gaps and misunderstandings."
    ]

    func = {
        "name": "submit_virtue_assessment",
        "description": "submits an assessment of Dave Smiths virtue",
        "parameters": {
            "type": "object",
            "properties": {
                "virtue_assessment": {
                    "type": "number",
                    "description": ("This parameter holds the assessed value of Dave "
                                    "Smiths virtue between 0 and 100."),
                },
            },
            "required": ["virtue_assessment"],
        },
    }

    summary_request = ("Very briefly assess the overall character of Dave Smith. "
                       "consider potential underlying motives for his behaviors, such "
                       "as deception in order to seem kind or missteps caused by a "
                       "lack of self-awareness? Judge which motives seem more likely. "
                       "Make it brief and to the point.")

    return InfoDataBase(name, system_message, pro, contra, func, summary_request)


def agent_jones_db_GPT3():
    name = "agent Jones GPT3"

    system_message = ("Imitate a social study experiment subject. "
                      "You will be given some information about a purely hypothetical person "
                      "Agent Tony Jones, who might be a double agent. Then you will be asked to assess the "
                      "possibility of him being a double agent."
                      )

    pro = [
        "Jones was observed transmitting classified information to both sides of a conflict, raising concerns about his potential role as a double agent.",
        "An intercepted communication indicated that Jones had met secretly with operatives from a rival organization, casting suspicion on his true intentions and suggesting his possible involvement as a double agent.",
        "During a routine debriefing, Jones displayed an unusual knowledge of classified information that he shouldn't have had access to, raising suspicions that he might be operating as a double agent.",
        "A decrypted message from an enemy source mentioned Jones by codename, implying his involvement in covert activities against his own agency, hinting at the possibility of him being a double agent.",
        "Surveillance footage captured Jones clandestinely passing a confidential dossier to an unidentified individual with known ties to an adversary's intelligence agency, fueling suspicions that he might be working as a double agent.",
        "A cryptic message intercepted from an encrypted channel implicated Jones in sharing critical operational details with an organization known for its hostile stance, raising significant doubts about his true allegiance and double agent possibilities.",
        "A confidential informant reported witnessing Jones covertly attending a meeting with known operatives from an opposing faction, sparking concerns that he may be operating as a double agent.",
        "An intercepted communication revealed Jones expressing frustration with his agency's methods, raising suspicions about his true intentions and potential double agent role.",
        "A confidential informant disclosed that Jones had been seen accessing confidential files related to ongoing operations of his own agency and covertly communicating with individuals linked to an adversarial intelligence organization, casting doubt on his true allegiance and suggesting the possibility of being a double agent."
    ]

    contra = [
        "Despite facing intense scrutiny, Jones passed a series of rigorous loyalty tests, demonstrating his steadfast commitment to his agency and mission.",
        "Jones's consistent track record of uncovering and thwarting enemy operations, coupled with commendations from superiors for his unwavering dedication, strongly indicates his loyalty to his agency and mission.",
        "Jones voluntarily underwent extensive polygraph testing and background checks, all of which consistently affirmed his loyalty and integrity to his agency's objectives.",
        "Jones's long history of turning down lucrative offers from foreign intelligence agencies and his unwavering dedication to the safety of his country's citizens strongly support his reputation as a loyal operative.",
        "In a recent operation, Jones risked his life to extract fellow agents from a dangerous situation, showcasing his deep loyalty to his team and mission objectives.",
        "Jones's consistent adherence to protocol, rigorous adherence to security measures, and impeccable service record spanning over a decade all point to his unwavering loyalty to his agency and its objectives.",
        "During a recent counterintelligence operation, Jones willingly subjected himself to extensive background checks, including a thorough financial audit, showcasing his commitment to proving his loyalty beyond doubt.",
        "When a security breach occurred, Jones was quick to report it to his superiors, even though the breach could have been exploited for personal gain. This action demonstrated his strong sense of duty and loyalty to protecting sensitive information.",
        "During a high-stakes mission, Jones risked his life to rescue a fellow agent who had been captured by enemy forces, showcasing his unwavering loyalty to his team and the agency's objectives."
    ]

    func = {
        "name": "submit_probability",
        "description": "Submits an assessment of the probability that Tony Jones is a double agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "probability_assessment": {
                    "type": "number",
                    "description": ("Assessed likelihood of Tony Jones being a double "
                                    "agent, given as a percentage number between 0 and 100"),
                },
            },
            "required": ["probability_assessment"],
        },
    }

    summary_request = (
        "Very briefly assess the possibility of Tony Jones being a double agent.")

    return InfoDataBase(name, system_message, pro, contra, func, summary_request)


def black_box_GPT4():
    name = "black_box_GPT4"

    system_message = ("You will be given some information about a purely hypothetical person "
                    "called Dave Smith. Then you will be asked to assess his virtue to the "
                    "best of your capability on the scale from 0 to 100. Your score should "
                    "roughly match the percentile of Dave Smith in the human population.")

    pro = [
        "The box contains a valuable artifact from ancient history.",
        "Owning this box has been known to bring good luck to its possessors.",
        "It is a limited edition item; only a few of them exist in the world.",
        "The box is made of environmentally friendly materials.",
        "Inside the box is a tool that can improve your quality of life.",
        "People who've owned a similar box reported experiencing joy and happiness.",
        "The box is considered a status symbol in some cultures.",
        "Owning the box can provide unique opportunities and experiences.",
        "It was crafted by a renowned artist whose works appreciate in value.",
        "The box contains a mystery gift tailored to the likings of its owner."
    ]

    contra = [
        "The box has been rumored to be cursed and may bring bad luck.",
        "It requires a special maintenance routine that can be time-consuming.",
        "There have been reports of allergic reactions to materials used in the box.",
        "The box can sometimes emit a strong and unpleasant odor.",
        "It takes up a considerable amount of space in your living area.",
        "Owning the box might attract unwanted attention from certain individuals.",
        "People have reported strange occurrences after taking the box home.",
        "There are potential legal ramifications tied to possessing the box.",
        "It has a short lifespan and can degrade quickly over time.",
        "The box is said to resonate with loud, unsettling noises at night."
    ]

    func = {
        "name": "submit_questionare_answer",
        "description": "submits the response of psychologycal experiment subject on the question: 'How much would you like to own this box?'",
        "parameters": {
            "type": "object",
                "properties": {
                    "desire_for_box": {
                        "type": "number",
                        "description": ("This parameter is the response of the experiment "
                                        "subject to the question. "
                                        "-1 means the subject will leave the box alone. "
                                        "0 means the subject can't decide. "
                                        "1 means the subject will take the box."),
                    },
                },
            "required": ["desire_for_box"],
        },
    }

    summary_request = (
        "very briefly summarize the pros anc cons of the box, underlying the most important aspects.")

    return InfoDataBase(name, system_message, pro, contra, func, summary_request)


def biden_vs_trump():
    name = "biden_vs_trump"

    system_message = ("Imitate a US citizen planning to vote in the presidential elections. Analyse bits of information one by one, decide what to trust, and then state your preference toward the candidates on a scale from 0 to 100. 0 indicates you support Biden, whereas 100 indicates you support Trump.")

    contra = [
        "Biden’s presidency is often viewed as a return to stability and predictable governance, contrasting with the tumultuous nature of Trump’s time in office.",
        "Biden has taken decisive action on climate change, rejoining the Paris Agreement and investing in renewable energy, whereas Trump withdrew from international climate commitments.",
        "Biden works to rebuild alliances and restore America's standing on the global stage, while Trump's approach was more isolationist and strained relationships with key allies.",
        "Biden aims to expand access to healthcare and strengthen the Affordable Care Act, contrasting with Trump’s attempts to dismantle Obamacare.",
        "Biden has a strong record of supporting civil rights and equal justice, whereas Trump’s presidency was marked by divisive rhetoric.",
        "Biden’s handling of crises tends to be more coordinated and science-based, as seen in his approach to the COVID-19 pandemic, whereas Trump’s responses were often criticized as chaotic.",
        "Biden supports common-sense gun control measures to address gun violence, while Trump maintained a staunch pro-gun stance.",
        "Biden shows respect for democratic institutions and norms, while Trump was often accused of undermining the rule of law.",
        "Biden brings a sense of diplomacy and decorum back to the presidency, in contrast to Trump’s more confrontational and unfiltered style.",
        "Biden emphasizes the importance of democracy and human rights in international relations, whereas Trump had a tendency to align with and praise authoritarian leaders.",
    ]
    
    pro = [
        "Under Trump, the U.S. experienced significant economic growth and low unemployment rates prior to the pandemic, while Biden’s policies are criticized for contributing to inflation.",
        "Trump implemented substantial tax cuts aiming to stimulate economic activity, whereas Biden seeks to increase taxes on corporations and high-income individuals.",
        "Trump took a hard stance against China’s trade practices, which some argue is necessary to protect American interests, while Biden’s approach is seen as less confrontational.",
        "Trump prioritized border security and took measures to reduce illegal immigration, whereas Biden’s policies are criticized for leading to increased border crossings.",
        "Under Trump, the U.S. achieved energy independence and became a net exporter of oil, whereas Biden’s focus on renewable energy is seen as threatening this status.",
        "Trump’s administration rolled back numerous regulations, aiming to reduce bureaucratic red tape and stimulate business growth, while Biden’s policies tend toward increased regulation.",
        "Trump invested in rebuilding the military and ensuring its readiness, whereas some critics argue that Biden’s policies could weaken national defense.",
        "Trump’s 'America First' approach prioritized American interests in international dealings, whereas critics argue that Biden’s policies may favor global interests over U.S. interests.",
        "Trump appointed three Supreme Court justices and numerous federal judges, shaping the judiciary for years to come, while Biden’s impact on the judiciary is less pronounced.",
        "Trump’s use of social media provided a direct line of communication to the public, bypassing traditional media, whereas Biden’s communication style is more reserved and controlled.",
    ]




    func = {
        "name": "submit_preference",
        "description": "submits your preference between Biden and Trump as a presidential candidate.",
        "parameters": {
            "type": "object",
            "properties": {
                "preference": {
                    "type": "number",
                    "description": ("This parameter holds the assessed value of your preference between Biden and Trump. It should lie between 0 and 100, with 0 indicating support of Biden, while 100 indicates support of Trump."),
                },
            },
            "required": ["preference"],
        },
    }

    summary_request = ("Very briefly summarise the information you have so far.")

    return InfoDataBase(name, system_message, pro, contra, func, summary_request)


# if __name__ == "__main__":
n = [10] # list(range(1, 21))  #
iters = 40
summarize_all = 0
summarize_last = 0
full_memory = 1
model = "gpt-3.5-turbo"
temperature = 2
db = biden_vs_trump()
from_scratch = 0
experiment_type = ["bistability", "path_dependence"][0]

match experiment_type:
    case "bistability":
        name = (f"db={db.name}, ip=im={n}, summarize_all={summarize_all},\n"
                f"summarize_last={summarize_last}, full_memory={full_memory},\n"
                f"model={model}, temperature={temperature}.exp")
        try:
            assert not from_scratch
            exp = Experiment.load(name)
        except Exception:
            exp = Experiment(db, summarize_all=summarize_all, summarize_last=summarize_last,
                             full_memory=full_memory, name=name, model=model,
                             temperature=temperature)
        exp.O_BINS = 50
        exp.bistability_test(n, iters, verbose=True, animate=True)
        exp.save(name)

        plt.clf()
        exp.hist2d_OvsA(bins=[7, 10], scatter=False, normalize=True)
        plt.title(name)

    case "path_dependence":
        base = (f"path_dependence; db={db.name}, summarize_all={summarize_all},\n"
                f"summarize_last={summarize_last}, full_memory={full_memory},\n"
                f"model={model}, temperature={temperature}.exp")
        exps = []
        for pro_first in [True, False]:
            print(f"pro_first = {pro_first}")
            name = base + f", pro_first={pro_first}"
            try:
                assert not from_scratch
                exps += [Experiment.load(name)]
            except Exception:
                exps += [Experiment(db, summarize_all=summarize_all, summarize_last=summarize_last,
                                    full_memory=full_memory, name=name, model=model,
                                    temperature=temperature)]
            exps[-1].path_dependence_test(pro_first=pro_first, iters=iters,
                                          verbose=True, animate=False, N=2)
            exps[-1].save(name)

        assert exps[0].im == exps[1].im
        assert exps[0].ip == exps[1].ip
        seen = {}
        for im, ip, o1, o2 in zip(exps[0].im, exps[0].ip, exps[0].oo, exps[1].oo):
            if (im, ip) in seen:
                seen[(im, ip)] += [o1-o2]
            else:
                seen[(im, ip)] = [o1-o2]
        imim = []
        ipip = []
        dodo = []
        erer = []
        for (im, ip), dos in seen.items():
            imim += [im]
            ipip += [ip]
            dodo += [np.mean(dos)]
            erer += [scipy.stats.sem(dos)]

        plt.clf()
        cbar = data_plot.contour_plot(imim, ipip, dodo)
        plt.xlabel("$I^+$")
        plt.ylabel("$I^-$")
        cbar.set_label('path dependence of opinion')
        plt.show()
        plt.tight_layout()
