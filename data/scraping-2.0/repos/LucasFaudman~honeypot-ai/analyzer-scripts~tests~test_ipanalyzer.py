from analyzerbase import *

from osintanalyzers.ipanalyzer import IPAnalyzer
from loganalyzers.cowrieloganalyzer import CowrieLogAnalyzer, Attack
from openaianalyzers.openaianalyzer import OpenAIAnalyzer
from main import AttackAnalyzer



class TestIPAnalyzer(TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.analyzer = AttackAnalyzer()
        cls.ia = IPAnalyzer()
        cls.ips = ['80.94.92.20']
        


    def test_init(self):
        self.assertIsInstance(self.ia, IPAnalyzer)
        #self.assertIsInstance(self.ia.attacks, dict)
        


    def test_get_data(self):
        data = self.ia.get_data(self.ips)

        # Ouput Structure
        self.assertIsInstance(data, dict)
        self.assertIsInstance(data[self.ips[0]], dict)

        #ISC
        self.assertIsInstance(data[self.ips[0]]['isc'], dict)
        self.assertIsInstance(data[self.ips[0]]['isc']['ip'], dict)
        self.assertIsInstance(data[self.ips[0]]['isc']['ip']['count'], int)
        self.assertIsInstance(data[self.ips[0]]['isc']['ip']['attacks'], int)

        #Whois
        self.assertIsInstance(data[self.ips[0]]['whois'], dict)
        self.assertIsInstance(data[self.ips[0]]['whois']['results'], dict)
        self.assertIsInstance(data[self.ips[0]]['whois']['results']['whois_text'], str)
        self.assertIsInstance(data[self.ips[0]]['whois']['results']['whois_list'], list)

        #Cybergordon
        
        self.assertIsInstance(data[self.ips[0]]['cybergordon'], dict)
        self.assertIsInstance(data[self.ips[0]]['cybergordon']['results'], dict)
        for priority in ['low', 'medium', 'high', "none", "error"]:
            if data[self.ips[0]]['cybergordon']['results'].get(priority):
                self.assertIsInstance(data[self.ips[0]]['cybergordon']['results'][priority], list)

        #Threatfox
        self.assertIsInstance(data[self.ips[0]]['threatfox'], dict)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'], list)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0], dict)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['date'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['ioc'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['ioc_url'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['ioc_data'], dict)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['ioc_data']['Confidence Level'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['ioc_data']['First seen'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['ioc_data']['Last seen'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['ioc_data']['Malware alias'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['ioc_data']['Threat Type'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['malware'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['malware_data'], dict)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['malware_data']['First seen'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['malware_data']['Last seen'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['malware_data']['Malpedia'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['malware_data']['Malware alias'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['malware_data']['Number of IOCs'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['malware_data']['Malpedia'], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['tags'], list)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['tags'][0], str)
        self.assertIsInstance(data[self.ips[0]]['threatfox']['results'][0]['reporter'], str)

        #Shodan
        self.assertIsInstance(data, dict)
        self.assertIsInstance(data[self.ips[0]], dict)
        self.assertIsInstance(data[self.ips[0]]['shodan'], dict)
        self.assertIsInstance(data[self.ips[0]]['shodan']['results'], dict)
        self.assertIsInstance(data[self.ips[0]]['shodan']['results']['ports'], dict)
        self.assertIsInstance(data[self.ips[0]]['shodan']['results']['ports'][0], dict)
        self.assertIsInstance(data[self.ips[0]]['shodan']['results']['ports'][0]['port'], int)
        self.assertIsInstance(data[self.ips[0]]['shodan']['results']['ports'][0]['protocol'], str)
        self.assertIsInstance(data[self.ips[0]]['shodan']['results']['ports'][0]['service'], str)
        self.assertIsInstance(data[self.ips[0]]['shodan']['results']['ports'][0]['version'], str)
        self.assertIsInstance(data[self.ips[0]]['shodan']['results']['ports'][0]['cpe'], list)
        self.assertIsInstance(data[self.ips[0]]['shodan']['results']['ports'][0]['cpe'][0], str)
        # self.assertIsInstance(data[self.ips[0]]['shodan']['results']['ports'][0]['cpe'][1], str)
        # self.assertIsInstance(data[self.ips[0]]['shodan']['results']['ports'][0]['cpe'][2], str)
        self.assertIsInstance(data[self.ips[0]]['shodan']['results']['general'], dict)



    # def test_check_shodan(self):
    #     shodan_data = self.ia.check_shodan(self.ips[0])
    #     self.assertIsInstance(shodan_data, dict)
    #     self.assertIsInstance(shodan_data['results'], dict)


    # def test_check_threatfox(self):
    #     tf_data = self.ia.check_threatfox(self.ips[0])
    #     self.assertIsInstance(tf_data, dict)
    #     self.assertIsInstance(tf_data['results'], list)

    # def test_check_whois(self):
    #     whois_data = self.ia.check_whois(self.ips[0])
    #     self.assertIsInstance(whois_data, dict)
    #     self.assertIsInstance(whois_data['results'], dict)
    
    # def test_check_isc(self):
    #     isc_data = self.ia.check_isc(self.ips[0])
    #     self.assertIsInstance(isc_data, dict)
    #     self.assertIsInstance(isc_data['results'], dict)

    # def test_check_cybergordon(self):
    #     cg_data = self.ia.check_cybergordon(self.ips[0])
    #     self.assertIsInstance(cg_data, dict)
    #     self.assertIsInstance(cg_data['results'], dict)

    