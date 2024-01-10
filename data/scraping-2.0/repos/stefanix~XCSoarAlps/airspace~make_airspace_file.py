# generate a xcsoar airspace file from multiple openaip.net files
# 1) download .aip files from openaip.net and place in aip_in
# 2) run this script
# 3) copy airspace_openaip_alps.txt to your XCSoarData folder
#    (may want to rename it if not alps)
# PREREQUISITES:  python2.7,  php7.1-cli, php7.1-xml

import os



def metacheck(airspace1, airspace2):
    # airspace class match
    if airspace1[0] == airspace2[0]:
        class_match = True
    else:
        class_match = False
    # airspace top match
    if airspace1[2] == airspace2[2]:
        top_match = True
    else:
        top_match = False
    # airspace bottom match
    if airspace1[3] == airspace2[3]:
        bottom_match = True
    else:
        bottom_match = False

    return class_match and top_match and bottom_match


def areacheck100pct(airspace1, airspace2):
    match = True
    # by line comparison
    if len(airspace1) == len(airspace2):
        for i in range(len(airspace1)):
            if i > 3 and airspace1[i] != airspace2[i]:
                match = False
    else:
        match = False
    return match


def areacheckpartial(airspace1, airspace2):
    match_count = 0
    for l in airspace1[4:]:
        for ll in airspace2[4:]:
            if l == ll:
                match_count += 1
    if  match_count > 0.9*len(airspace2): #90% match
        return True
    else:
        return False


def printairspace(airspace1, airspace2):
    print "- %s - %s - %s - %s" % (airspace2[1][3:].strip(), airspace2[0][3:].strip(), airspace2[2][3:].strip(), airspace2[3][3:].strip())
    print "= %s - %s - %s - %s" % (airspace1[1][3:].strip(), airspace1[0][3:].strip(), airspace1[2][3:].strip(), airspace1[3][3:].strip())


def preserve(airspace_name, airspace_current, airspaces, airspaces_duplies):
    if airspace_name not in airspaces_duplies:
        airspaces_duplies[airspace_name] = []
    else:
        # duplie check
        got_identical = False
        # in duplies
        for space in airspaces_duplies[airspace_name]:
            if areacheck100pct(airspace_current, space) and metacheck(airspace_current, space):
                # printairspace(airspace_current, space)
                got_identical = True

        if got_identical:
            # identical already there -> ignore
            return

    airspaces_duplies[airspace_name].append(airspace_current)



def filter_redundant_airspaces(input_file, output_file):
    airspaces = {}
    airspaces_duplies = {}
    airspace_name_duplies = {}
    airspace_current = []

    name_duplies1 = 0
    data_duplies1 = 0
    data_duplies2 = 0
    name_duplies_diffmeta1 = 0
    name_duplies_diffmeta2 = 0
    name_duplies_diffairspace = 0

    fp = open(input_file, "r")
    for line in fp:
        if line[0] == "*":
            continue
        elif line.strip() == "":
            if airspace_current:
                airspace_name = airspace_current[1].strip()
                if airspace_name not in airspaces:
                    airspaces[airspace_name] = airspace_current
                else:
                    name_duplies1 += 1

                    if areacheck100pct(airspace_current, airspaces[airspace_name]):
                        if metacheck(airspace_current, airspaces[airspace_name]):
                            # same meta, 100% airspace match -> ignore
                            data_duplies1 += 1
                        else:
                            # diff meta, 100% airspace match -> preserve
                            preserve(airspace_name, airspace_current, airspaces, airspaces_duplies)
                            name_duplies_diffmeta1 += 1
                    else:
                        # diff airspace -> preserve
                        preserve(airspace_name, airspace_current, airspaces, airspaces_duplies)
                        name_duplies_diffairspace += 1



            airspace_current = []
        else:
            airspace_current.append(line)

    # for k in airspaces.keys():
    #     print k

    print "-----"
    print "duplies total: %s" % name_duplies1
    print "duplies, 100pct airspace match: %s (removed)" % data_duplies1
    # print "duplies, 90pct airspace match: %s (preserved)" % data_duplies2
    print "duplies, diff meta, 100pct airspace match: %s (preserved)" % name_duplies_diffmeta1
    # print "duplies, diff meta, 90pct airspace match: %s (preserved)" % name_duplies_diffmeta2
    print "duplies, diff airspace: %s (preserved)" % name_duplies_diffairspace
    print "-----"

    fp.close()

    fpout = open(output_file, "w")
    for k in airspaces.keys():
        space = airspaces[k]
        for lines in space:
            for line in lines:
                fpout.write(line)
        fpout.write("\n")

    for k in airspaces_duplies.keys():
        spaces = airspaces_duplies[k]
        for space in spaces:
            for lines in space:
                for line in lines:
                    fpout.write(line)
            fpout.write("\n")

    fpout.close()


print "converting all .aip files in 'aip_in' ..."
os.system("php airspace-converter/aip2openair.php")
print "concatinating all openair files into airspace_openaip_alps_raw.txt ..."
os.system("cat openair_out/*.txt > airspace_openaip_alps_raw.txt")
print "filtering redundant airspaces ..."
filter_redundant_airspaces("airspace_openaip_alps_raw.txt", "airspace_openaip_alps.txt")
print "airspace_openaip_alps.txt generated, done!"
