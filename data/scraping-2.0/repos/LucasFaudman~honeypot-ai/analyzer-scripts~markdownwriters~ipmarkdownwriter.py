from analyzerbase import *
from markdownwriters.markdownwriterbase import *


from osintanalyzers.ipanalyzer import IPAnalyzer
from loganalyzers.cowrieloganalyzer import CowrieLogAnalyzer, Attack
from openaianalyzers.openaianalyzer import OpenAIAnalyzer

from .visualizer import CounterGrapher


class IPAnalyzerMarkdownWriter(MarkdownWriter):
    """Writes markdown for IPAnalyzer ipdata"""

    def prepare(self):
        
        #self.md += h1("What do you know about the attacker?")
        self.md_editors.append(self.add_osint_header)
        self.md_editors.append(self.add_ip_locations)
        self.md_editors.append(self.add_cybergordon)
        self.md_editors.append(self.add_shodan)
        self.md_editors.append(self.add_threatfox)
        self.md_editors.append(self.add_isc)
        self.md_editors.append(self.add_whois)
        

    def add_osint_header(self, md, attack: Attack):
        md += h1("What do you know about the attacker?")
        md += attack.answers['osint_summary'] + "\n"

        return md

    def add_isc_ip_locations(self, md, attack: Attack):
        data = attack.ipdata

        ip_loc_md = h3("IP Locations")
        
        location_data = [(ip, 
                        data[ip]["isc"]["results"]["ascountry"],
                        data[ip]["isc"]["results"]["as"],
                        data[ip]["isc"]["results"]["asname"],
                        data[ip]["isc"]["results"]["network"]
                        ) for ip in data if data[ip]["isc"].get("results")]

        location_data.sort(key=lambda x: 
                           (data["counts"]["isc"]["ascountry"].get(x[1]),
                            data["counts"]["isc"]["asname"].get(x[3]),
                            data["counts"]["isc"]["network"].get(x[4])), 
                            reverse=True)
        
        ip_loc_md += table(['IP Address', 'Country', "AS", "AS Name", "Network"], location_data)
        
        md += collapseable_section(ip_loc_md,
                                    "IP Locations",
                                    header_level=2,
                                    )
        
        return md
    


    def add_ip_locations(self, md, attack: Attack):
        data = attack.ipdata

        ip_loc_md = h3("IP Locations Summary")
        ip_loc_md += attack.answers['ip_locations_summary'] + "\n"


        counts = data["counts"]
        summary = [
            f"This attack involved {code(attack.num_uniq_ips)} unique IP addresses. "
            f"{code(attack.num_uniq_src_ips)} were source IPs."
            f"{code(attack.num_uniq_cmdlog_ips)} unique IPs and {code(attack.num_uniq_cmdlog_ips)} unique URLS were found in the commands."
            f"{code(attack.num_uniq_malware_ips)} unique IPs and {code(attack.num_uniq_malware_ips)} unique URLS were found in malware."
        ]
        for key in ["Country", "City", "ISP", "Organization", "ASN"]:
            if not counts['shodan'].get(key):
                continue

            summary.append(f"The most common {bold(key)} of origin was {code(counts['shodan'][key].most_common(1)[0][0])}, "
                        f"which was seen {code(counts['shodan'][key].most_common(1)[0][1])} times.")
            
        summary.append(f"The most common {bold('network')} of origin was {code(counts['isc']['network'].most_common(1)[0][0])}, "
                        f"which was seen {code(counts['isc']['network'].most_common(1)[0][1])} times.")

        ip_loc_md += unordered_list(summary)

        location_data = [
            (ip, 
            data[ip]["shodan"]["results"]["general"]["Country"],
            data[ip]["shodan"]["results"]["general"]["City"],
            data[ip]["shodan"]["results"]["general"]["ISP"],
            data[ip]["shodan"]["results"]["general"]["Organization"],
            #data[ip]["isc"]["results"]["asname"],
            data[ip]["shodan"]["results"]["general"]["ASN"],
            data[ip]["isc"]["results"]["network"]
            ) for ip in data \
                if data[ip]["isc"].get("results") \
                and data[ip]["shodan"].get("results")
        ]

        location_data.sort(key=lambda x: 
                           (counts["shodan"]["Country"].get(x[1]),
                            counts["shodan"]["City"].get(x[2]),
                            counts["shodan"]["ISP"].get(x[3]),
                            counts["shodan"]["Organization"].get(x[4]),
                            counts["shodan"]["ASN"].get(x[5]),                            
                            counts["isc"]["network"].get(x[5])
                            
                            ), 
                            reverse=True)
        
        location_headers = ['IP Address', 'Country', "City", "ISP", "Organization",  "ASN", "Network"]
        ip_loc_md += table(location_headers, location_data)
        
        md += collapseable_section(ip_loc_md,
                                    "IP Locations",
                                    header_level=2,
                                    )
        
        return md
    


    def add_isc(self, md, attack: Attack):
        data = attack.ipdata
        
        isc_counts = data["counts"]["isc"]

        if len(data) == 1:
            sharing_url = list(data.values())[0]["isc"]["sharing_link"]
            sharing_link = link(sharing_url, sharing_url)
        else:
            sharing_link = link("https://isc.sans.edu/ipinfo/", "https://isc.sans.edu/ipinfo/")

        isc_md = h3("Internet Storm Center (ISC) " + sharing_link)
        
        isc_md += attack.answers['isc_summary'] + "\n"

        isc_data = []
        #threatfeed_data = []
        #threatfeed_headers = ["IP Address", "Total Reports"] + sorted(isc_counts["threatfeeds"].keys())
        
        num_ips_with_reports = 0
        max_reports = (None, 0)
        max_targets = (None, 0)
        first_seen = (None, '')
        most_recent = (None, '')

        for ip in data:
            if ip == "counts":
                continue

            ip_results = data[ip].get("isc")["results"]
            
            if ip_results.get("count"):
                num_ips_with_reports += 1

                if ip_results["count"] > max_reports[1]:
                    max_reports = ip, ip_results["count"]
                
                if ip_results["attacks"] > max_targets[1]:
                    max_targets = ip, ip_results["attacks"]


                if not first_seen[1] or ip_results["mindate"] < first_seen[1]:
                    first_seen = ip, ip_results["mindate"]

                if not most_recent[1] or ip_results["maxdate"] > most_recent[1]:
                    most_recent = ip, ip_results["maxdate"]

            isc_data.append((
                ip, 
                ip_results["count"] or 0,
                ip_results["attacks"] or 0,
                ip_results["mindate"],
                ip_results["maxdate"],
                ip_results["updated"],
                        ))
            
            #threatfeed_data.append((
            #    ip,
            #    sum(ip_results["threatfeeds"].values()),

                                    

        summary = [
            f"{code(num_ips_with_reports)} of the {code(len(data) - 1)} unique source IPs have reports on the Internet Storm Center (ISC).",
            f"{code(isc_counts['count']['total'])} total attacks were reported.",
            f"{code(isc_counts['attacks']['total'])} unique targets were attacked.",
            f"The IP address with the {bold('most reports')} was {code(max_reports[0])} with {code(max_reports[1])} reports.",
            f"The IP address with the {bold('most targets')} was {code(max_targets[0])} with {code(max_targets[1])} targets.",
            f"The {bold('first report')} was on {code(first_seen[1])} from {code(first_seen[0])}.",
            f"The {bold('most recent')} was on {code(most_recent[1])} from {code(most_recent[0])}.",

        ]

        isc_md += unordered_list(summary)


        isc_data.sort(
            key=lambda x: (x[1], x[2], x[3], x[4]),
            reverse=True
        )

        isc_headers = ['IP Address', 'Total Reports', "Targets", "First Report", "Last Report", "Update Time"]
        isc_md += table(isc_headers, isc_data)

        
        for key in isc_counts.keys():
            if key.startswith("as") or key.startswith("cloud") or key in ["network", "threatfeeds"]:
                isc_md += most_common_table(key, isc_counts[key], n=10, style_fn=code, header_level=4)
        
        

        md += collapseable_section(isc_md,
                                    "Internet Storm Center (ISC)",
                                    header_level=2,
                                    )

        #TODO ADD THREATFEEDS

        return md    



    def add_whois(self, md, attack: Attack):
        data = attack.ipdata
        whois_md = h3("Whois Results Summary")

        
        for ip in data:
            if ip == "counts":
                continue
            whois_data = data[ip].get("whois")
            if not whois_data or whois_data.get("error"):
                #TODO Backup whois from isc
                continue

            sharing_link = link(whois_data["sharing_link"], whois_data["sharing_link"])
            
            ip_md = h3(f"Whois data for: {ip} " + sharing_link)
            ip_md += codeblock(whois_data["results"].get("whois_text", whois_data["error"]))
            whois_md += collapseable_section(ip_md,
                                       f"Whois data for: {ip}", #+ sharing_link,
                                       header_level=3,
                                       )

        md += collapseable_section(whois_md,
                                    "Whois",
                                    header_level=2
                                    )
        
        return md



    def add_cybergordon(self, md, attack: Attack):
        data = attack.ipdata
        cybergordon_counts = data["counts"]["cybergordon"]

        cybergordon_md = h3("CyberGordon Results")
        
        
        all_engines = sorted([key for key in cybergordon_counts.keys() if cybergordon_counts[key]['alerts'] > 0],
                             key=lambda engine: 
                             (int(engine.split()[0].strip("[E]")), # Sort by Engine #
                              cybergordon_counts[engine]['alerts'], # Then total alerts 
                              cybergordon_counts[engine]['high'], # Then by priority
                              cybergordon_counts[engine]['medium'],
                              cybergordon_counts[engine]['low']
                              ), 
                                            )
                        
        combined_headers = ["IP Addresss", "Alerts High | Med | Low"] + all_engines
        combined_results = []
        
        max_high_alerts = (None, 0)

        for ip in data:
            if ip == "counts":
                continue

            cybergordon_data = data[ip].get("cybergordon")
            
            if not cybergordon_data or cybergordon_data.get("error"):
                #md += codeblock(cybergordon_data)
                continue
            

            cybergordon_results = cybergordon_data["results"]
            
            combined_row = [
                ip,
                f"{code(len(cybergordon_results.get('high', {})))}"
                f" | {code(len(cybergordon_results.get('medium', {})))}"
                f" | {code(len(cybergordon_results.get('low', {})))}",
                
                ] + [None for _ in all_engines]
                
            sharing_link = link(cybergordon_data["sharing_link"], cybergordon_data["sharing_link"])
            
            ip_md = h3(f"Cybergordon results for: {ip} " + sharing_link)
            cybergordon_table_data = []

            ip_alerts = 0
            for priority in ["high", "medium", "low"]:
                for entry in cybergordon_data["results"].get(priority, []):
                    cybergordon_table_data.append((entry["engine"], entry["result"], entry["url"]))
                    
                    combined_row[combined_headers.index(entry["engine"])] = collapsed(
                        f"{code(entry['result'])}",
                        priority,
                        code,
                    )


                    ip_alerts += 1
                    if priority == "high" and ip_alerts > max_high_alerts[1]:
                        max_high_alerts = ip, ip_alerts

            cybergordon_table_headers = ['Engine', 'Results', "Url"]
            ip_md += table(cybergordon_table_headers, cybergordon_table_data)
            cybergordon_md += collapseable_section(ip_md,
                                       f"Cybergordon results for: {ip}", #+ sharing_link,
                                       header_level=3,
                                       )
        
            combined_results.append(combined_row)

        combined_summary_md = h3("CyberGordon Results Summary")
        combined_summary_md += attack.answers['cybergordon_summary'] + "\n"

        alert_counts = defaultdict(int)
        for engine in all_engines:
            for key in ["alerts", "high", "medium", "low"]:
                alert_counts[key] += cybergordon_counts[engine][key]    

        summary = [
            f"{code(alert_counts['alerts'])} total alerts were found across all engines.",
            f"{code(alert_counts['high'])} were {bold('high')} priority. ",
            f"{code(alert_counts['medium'])} were {bold('medium')} priority. ",
            f"{code(alert_counts['low'])} were {bold('low')} priority. ",
            #TODO engine triggered most results
            f"The IP address with the {bold('most high priority alerts')} was {code(max_high_alerts[0])} with {code(max_high_alerts[1])} alerts.",
        ]

        combined_results.sort(
                                # Sort the combined results by the number of high,med,low alerts
            key=lambda x: tuple(map(int, x[1].replace('`','').split(" | "))), 
            reverse=True)

        combined_summary_md += unordered_list(summary)
        combined_summary_md += table(combined_headers, combined_results)

        md += collapseable_section(
            combined_summary_md + cybergordon_md,
            "CyberGordon",
            header_level=2
            )

        return md
    


    def add_shodan(self, md, attack: Attack):
        data = attack.ipdata
        shodan_counts = data["counts"]["shodan"]

        shodan_md =  h3("Shodan Results")
        summary_md = h3("Shodan Results Summary")

        summary_md += attack.answers['shodan_summary'] + "\n"

        ips_with_shodan = 0
        max_open_ports = (None, 0)
        #max_vulns = (None, 0)

        combined_headers = ["IP Addresss", "# Open Ports"] + sorted(shodan_counts["ports"].keys(), key=int)
        combined_results = []

        for ip in data:
            if ip == "counts":
                continue
            
            shodan_data = data[ip].get("shodan")
            if not shodan_data or shodan_data.get("error"):
                #md += codeblock(shodan_data)
                continue

            ips_with_shodan += 1

            sharing_link = link(shodan_data["sharing_link"], shodan_data["sharing_link"])
            
            ip_md = h3(f"Shodan results for: {ip} " + sharing_link)

            headers = list(shodan_data["results"]["general"].keys())
            shodan_general_values = list(shodan_data["results"]["general"].values())
            ip_md += table(headers, [shodan_general_values])
            
            ip_md += h4("Open Ports")
            all_ports_table_headers = ["Port", "Protocol", "Service", "Update Time"]
            all_ports_data = []
            ports_md = ""

            combined_row = [ip, f"{code(len(shodan_data['results']['ports']))}"] + ["-" for _ in shodan_counts["ports"]]

            for port, entry in shodan_data["results"]["ports"].items():
                all_ports_data.append((port, entry["protocol"], entry["service_name"], entry["timestamp"]))
            
                port_label = f"Port {port} ({entry['protocol']}): {entry['service_name']}"
                port_md = h4(port_label)
                port_md += collapseable_section(codeblock(entry["service_data_raw"]), 
                                                "Raw Service Data for " + port_label, 
                                                header_level=4, 
                                                #blockquote=True
                                                ) 
                port_md += table(["Key", "Value"], [(key.strip(), entry["service_data"][key]) for key in entry["service_data"]])
                ports_md += port_md

                combined_row[combined_headers.index(port)] = entry["service_name"]
                
            
            combined_row[combined_headers.index("# Open Ports")] = collapsed(
                md_join(shodan_data["results"]["ports"].keys(), code),
                f"{code(len(all_ports_data))}",
            )

            combined_results.append(combined_row)

            ip_md += table(all_ports_table_headers, all_ports_data)

            if len(all_ports_data) > max_open_ports[1]:
                max_open_ports = ip, len(all_ports_data)
            
            

            #ip_md += h4("Vulnerabilities")
            # TODO ADD SHODAN VULN DATA
            shodan_md += collapseable_section(ip_md + ports_md,
                                        f"Shodan results for: {ip}", #+ sharing_link,
                                        header_level=3,
                                        )

        combined_results.sort(key=lambda x: len(data[x[0]]["shodan"]["results"]["ports"]), reverse=True)
                              #int(x[1].replace('`','')))
        combined_table = table(combined_headers, combined_results)

        name_key_pairs = {
            "open port": "ports",
            "protocol": "protocol",
            "service name": "service_name",
            "service signature": "sig",

        }

        for key in shodan_counts.keys():
            if key not in list(name_key_pairs.values()) + ["service_data_raw"]:
                
                name_key_pairs[key] = key
            

        mc_tables = []
        summary =[]
        n = 10
        style_fn = code
        header_level = 4

        for name, key in name_key_pairs.items():
            if not shodan_counts.get(key):
                continue

            summary.append(f"The most common {bold(name)} was {code(shodan_counts[key].most_common(1)[0][0])}, "
                        f"which was seen {code(shodan_counts[key].most_common(1)[0][1])} times.")
            
            mc_tables.append(
                most_common_table(
                    name.title() if not name.isupper() else name, 
                    shodan_counts[key], 
                    n, 
                    style_fn, 
                    header_level)
                )
            
        summary.append(f"The IP address with the {bold('most open ports')} was {code(max_open_ports[0])} with {code(max_open_ports[1])} open ports.")


        summary_md += nested_list(summary) 
        summary_md += combined_table
        summary_md += '\n\n'.join(mc_tables)
        

        md += collapseable_section(summary_md + shodan_md,
                                    "Shodan",
                                    header_level=2,
                                    )

        return md



    def add_threatfox(self, md, attack: Attack):
        data = attack.ipdata

        threatfox_md = h3("ThreatFox Results Summary")
        threatfox_md += attack.answers['threatfox_summary'] + "\n"

        #TODO ADD SUMMARY TABLE

        for ip in data:
            if ip == "counts":
                continue

            threatfox_data = data[ip].get("threatfox")
            if not threatfox_data or threatfox_data.get("error") or not threatfox_data.get("results"):
                #md += codeblock(threatfox_data)
                continue

            sharing_link = link(threatfox_data["sharing_link"], threatfox_data["sharing_link"])
            ip_md = h3(f"Threat Fox results for: {ip} " + sharing_link)
            
            
            threatfox_ioc_table_values = []
            threatfox_malware_table_values = []

            for ioc in threatfox_data["results"]:
                ioc_link = link(ioc["ioc"], ioc["ioc_url"])
                malware_link = link(ioc["malware"], ioc["malware_url"])

                ioc_data = ioc["ioc_data"]
                threatfox_ioc_table_values.append((ioc_link, ioc_data["Threat Type"], malware_link, ioc_data["Malware alias"], ioc_data["Confidence Level"], ioc_data["First seen"], ioc_data["Last seen"]))

                malware_data = ioc["malware_data"]
                threatfox_malware_table_values.append((malware_link, malware_data["Malware alias"], malware_data["Number of IOCs"], malware_data["First seen"], malware_data["Last seen"], malware_data["Malpedia"]))


            if threatfox_ioc_table_values:
                ip_md += h4(f"ThreatFox IOCS (Indicators of Compromise) Found for {ip}")            
                threatfox_ioc_table_headers = ["IOC", "Threat Type", "Malware", "Malware Alias", "Confidence Level", "First Seen", "Last Seen"]            
                ip_md += table(threatfox_ioc_table_headers, threatfox_ioc_table_values)

            if threatfox_malware_table_values:
                ip_md += h4(f"ThreatFox Malware Found for {ip}")
                threatfox_malware_table_headers = ["Malware", "Aliases", "Number of IOCs", "First Seen", "Last Seen", "Malpedia URL"]
                ip_md += table(threatfox_malware_table_headers, set(threatfox_malware_table_values))

            threatfox_md += collapseable_section(ip_md,
                                        f"ThreatFox results for: {ip}", #+ sharing_link,
                                        header_level=3,
                                        )

        md += collapseable_section(threatfox_md,
                                    "ThreatFox",
                                    header_level=2,
                                    )
        return md


