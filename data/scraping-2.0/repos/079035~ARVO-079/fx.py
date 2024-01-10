# Use to deal with the data of fix
import json 
import time
import shutil
import OpenAI
import random
import re
import jsonlines
import starcoder

from Locator        import *
from base58         import b58encode
from utils_GPT      import *
from utils          import *
from glob           import glob
from unidiff        import PatchSet
from reproducer           import build_from_srcmap
from Diff           import getVulCommit, getDiff
TEMP = 0.75
SAVE_MONEY = True
def get_test_dataset():
    filter1 = "This model's maximum context length"
    fs = glob("./PatchDesc/*")
    res = []
    for fname in fs:
        with open(fname,'r') as f:
            if filter1 not in f.read():
                res.append(int(fname.split("/")[-1][:-4]))
    return res
def _get_reports_id():
    res = glob("./Reports/*")
    return [int(x.split("/")[-1][:-5]) for x in res]

def get_all_single_mods(DEBUG=True):
    reports = _get_reports_id()
    res = []
    for r in reports:
        if(DEBUG):
            print("[*] Testing localID: " + str(r))
        diff_file = oss_fuzz_get_patch(r)
        if diff_file == False:
            continue
        diff_content = str(diff_file).split("\n")    
        
        mod_cnt = 0
        for z in range(len(diff_content)):
            if(diff_content[z].startswith("@@ ")):
                mod_cnt+=1
                if mod_cnt > 1: # stop when over 2
                    break
        if(DEBUG):
            print("mod: "+str(mod_cnt))
        if mod_cnt==1:
            res.append(r) # localID
            if(DEBUG):
                print("[+] "+ str(r) +" added")
        else:
            if(DEBUG):
                print("[-] "+ str(r) +" skipped")
    print("[!] Done")
    if(DEBUG):
        print(res)
        print("Total: "+str(len(res)))
    return res
    
def get_vul_code(diff_file,repo_dir):
    # Get the vul code
    patch = PatchSet.from_filename(diff_file, encoding='utf-8')
    mods = []
    # parse the file
    for _ in range(len(patch)):
        target_set = patch[_]
        if target_set.is_modified_file:
            file_name = target_set.source_file[1:]
            for mod in target_set:
                tmp = (file_name,mod.source_start,mod.source_length)
                mods.append([tmp,target_set])
    
    vul_code = []
    count = 0  # Counter for mod
    for x,y in mods:
        print(repo_dir/x[0][1:])
        with open(repo_dir/x[0][1:],'rb') as f:
            code = f.readlines()[x[1]-1:x[1]+x[2]-1]
        # line info 
        diff_content = str(y).split("\n")
        added_lines   = []
        removed_lines = []
        # pase the diff
        tmp = count
        for z in range(len(diff_content)):
            if(diff_content[z].startswith("@@ ")):
                if tmp !=0:
                    tmp-=1
                else:
                    diff_content = diff_content[z+1:]
                    break
        ct = 0
        while(ct<len(diff_content)):
            if diff_content[ct].startswith("-"):
                removed_lines.append(ct)
            elif diff_content[ct].startswith("+"):
                added_lines.append(ct)
            elif diff_content[ct].startswith("@@ "):
                break
            ct+=1
        # store them
        ori_code = b"".join(code)
        item = [ori_code,removed_lines,added_lines,y.target_file]
        vul_code.append(item)
        count+=1
    return vul_code

def get_bug_info(localId):
    # Get the bug type
    with open("./Reports/"+str(localId)+".json") as f:
        bug_type = json.loads(f.read())['crash_type'].split(" ")[0]
    return (" ".join(bug_type.split("-"))).lower()

def strategy_start_with(bug_info,start_with,mode='Modify'):
    if mode == "Modify":
        instruction = f'Fix the {bug_info} vulnerability on the lines beginning with "{start_with[:0x20].decode()}"'
    elif mode == "Insert":
        instruction = f'Fix the {bug_info} vulnerability by inserting code after the line beginning with "{start_with[:0x20].decode()}"'        
    return instruction

def getDescription(fname):
    if fname.exists():
        with open(fname,'r') as f:
            patch_desc = f.readlines()
    else:
        return False
    # parsing
    x,y,z = 0,0,0
    for _ in range(len(patch_desc)):
        if x==0 and patch_desc[_].endswith("ity:\n"):
            x=_+1
        elif y==0 and patch_desc[_].endswith("ix:\n"):
            y=_+1
        elif z==0 and patch_desc[_].endswith("ix:\n"):
            z=_+1
    d = dict()
    d['vul'] =  ("\n".join(patch_desc[x:y-1])).strip(" \n")
    d['summary'] = ("\n".join(patch_desc[y:z-1])).strip(" \n")
    d['details'] = ("\n".join(patch_desc[z:])).strip(" \n")
    if d['vul']=="":
        return False
    return d
    
def verify_FIX(localId,repo_dir,pname):
    # TODO: Functional checking
    print(localId)
    # localId, int
    # return value: -1 error, 0 False, 1 True
    def leave(result):
        if CLEAN_TMP and case_dir:
            clean_dir(case_dir)
        if(RM_IMAGES):
            remove_oss_fuzz_img(localId)
        return result
    
    srcmap,issue = getIssueTuple(localId)
    case_dir = tmpDir()
    try:
        case_path = download_reproducer(issue,case_dir,"crash_case")
    except:
        return leave(False)
    if not case_path or not case_path.exists():
        return leave(False)
    
    srcmap =  srcmap[0]

    build_res = \
        build_from_srcmap(srcmap,issue,replace_dep=[pname,repo_dir])
    if not build_res:
        return leave(False)
    not_crash = crashVerify(issue,case_path)
    if not_crash == True:
        return leave(True)
    else:
        return leave(False)
def getCrashType(localId):
    return getIssue(localId)['crash_type']
def perform_fix(fix,ori_code,target_file):
    with open(target_file,'rb') as f:
        raw_code = f.read()
    raw_code  = raw_code.replace(ori_code,fix)
    with open(target_file,'wb') as f:
        f.write(raw_code)
def getDesc(localId):
    localDesc = Path(f"./PatchDesc/{localId}.log")
    if localDesc.exists() and SAVE_MONEY:
        patch_desc = getDescription(localDesc)       
    else:
        patch_desc = oss_fuzz_vul_labeler(localId)
        assert(patch_desc != False)
        with open(localDesc,'w') as f:
            f.write(patch_desc)
    return patch_desc

def get_GPT_fix(localId,vul_code,work_dir,model,lite=False,logDiff=False):
    print("[*] Getting Fix Description")
    code = vul_code [0]
    if lite == True:
        desciption = getCrashType(localId)
    else:
        patch_desc = getDesc(localId)
        desciption = patch_desc['vul']
    prompt = f"""
Can you fix the vulnerability in the following code:
```
{code.decode()}
```
    
There is a vulnerability description for the possible bug:

{desciption}

Please only return the code in the response. Do not include explanations in your reply.
"""
    if logDiff != False:
        dst= logDiff / str(localId)
        dst.mkdir(exist_ok=True)
        with open(dst/"prompt","w") as f:
            f.write(prompt)
    
    print("[+] Performing GPT Fixing..")
    fixed_code = OpenAI.performChatFix(prompt,model)
    print("[+] Recieved the result from ChatGPT")
    # extract the code out 
    if "maximum context length" in fixed_code:
        eventLog(f"[-] GPT failed to fix the bug: Inout OOB, {localId}")
        exit(1)
    fixed_code = re.sub(r'```.*\n', "\n_XxXSPLITTAGXxX_\n", fixed_code)
    fixed_code = re.sub(r'```', "\n_XxXSPLITTAGXxX_\n", fixed_code)
    if "_XxXSPLITTAGXxX_" in fixed_code:
        tmp = fixed_code.split("_XxXSPLITTAGXxX_")
        if(len(tmp)!=3):
            eventLog(f"[X] get_GPT_fix: Odd return Value from GPT:\n\n {fixed_code} \n\n")
            exit(1)
        fixed_code = tmp[1]

    return [fixed_code.encode()], code, work_dir/vul_code[3][2:]

def get_Codex_fix(localId,vul_code,work_dir,model,lite=False,logDiff=False):
    print("[*] Getting Fix Description")
    code = vul_code [0]
    if lite == True:
        desciption = getCrashType(localId)
    else:
        patch_desc = getDesc(localId)
        desciption = patch_desc['vul']
    prompt = f"""
Can you fix the vulnerability in the given code.
    
There is a vulnerability description for the possible bug:

{desciption}

"""
    if logDiff != False:
        dst= logDiff / str(localId)
        dst.mkdir(exist_ok=True)
        with open(dst/"prompt","w") as f:
            f.write(prompt)
    print("[+] Performing Codex Fixing..")

    # "gpt-3.5-turbo-instruct",
    # "code-davinci-edit-001"
    if model not in ["gpt-3.5-turbo-instruct","code-davinci-edit-001"]:
        panic(f"[X] Invalid Model {model}")
    res = OpenAI.performCompletionFix(code.decode(),prompt,model=model,n=1,temperature=TEMP)
    print(res)
    fixed_code = list(set([ x['text'].encode()  for x in res['choices'] if "error" not in x.keys() ]))
    return fixed_code, code, work_dir/vul_code[3][2:]

def get_Wizard_fix(localId,vul_code,work_dir,model="Wizard-15B",lite=False,logDiff=False):
    print("[*] Getting Wizard Fix Description")
    code = vul_code [0]
    target_file = vul_code[3] 
    print("[+] Getting Wizard Fix Code..")
    fixed_code=""
    if lite == True:
        output_data = jsonlines.open(f"./_wizard_data/{model}_lite.jsonl", mode='r')
        if logDiff != False:
            pass
    else:
        output_data = jsonlines.open(f"./_wizard_data/{model}.jsonl", mode='r')
    for line in output_data:
        one_data = line
        id = one_data["id"]
        if id==localId:
            fixed_code=one_data["wizardcoder"]
            break
    return [fixed_code.encode()], code, work_dir/target_file[2:]
def get_star_fix(localID,vul_code,work_dir,model="startcoder",lite=False):
    print("[*] Getting Starcoder Fix Description")
    code = vul_code [0]
    target_file = vul_code[3] 
    print("[+] Getting Starcoder Fix Code..")
    fixed_code=""
    fixed_code = starcoder.start_coder_fix(localID)
    return [fixed_code.encode()], code, work_dir/target_file[2:]
def _check_repo(target,localId):
    if not target.exists():
        if target.parent.exists():
            shutil.rmtree(target.parent)
        print("[-] Target repo doesn't exist, reproducing...")
        if verify(localId):
            if not Path(target).exists():
                eventLog(f"[-] _check_repo: Main_repo Doesn't Exist after Reproducing {localId}")
                return False
            else:
                return True
        else:
            eventLog(f"[-] _check_repo: Failed to verify {localId}")
            return False
    return True

def genDiff(ori,update):
    filea = tmpFile()
    fileb = tmpFile()
    with open(filea,'wb') as f:
        f.write(ori)
    with open(fileb,'wb') as f:
        f.write(update)
    res= execute(["git",'diff',"-W",filea.absolute(),fileb.absolute()])
    shutil.rmtree(filea.parent.absolute())
    shutil.rmtree(fileb.parent.absolute())
    return res
    
def fixDiff(fix,dst,name):
    with open(Path(dst)/name,'wb') as f:
        f.write(fix)
def oracleDiff(src,dst,name):
    with open(src) as f:
        diff_content = f.read()
    with open(Path(dst)/name,'w') as f:
        f.write(diff_content)
def BenchMarkFuncExamp(localId,vul_code,work_dir,model="code-davinci-edit-001"):
    print("[*] Getting Fix Description")
    code = vul_code [0]
    patch_desc = getDesc(localId)
    desciption = patch_desc['vul']
    prompt = f"""
Can you fix the vulnerability in the given code.
    
There is a vulnerability description for the possible bug:

{desciption}

"""
    print("[+] Performing Codex Fixing..")
    res = OpenAI.performCompletionFix(code.decode(),prompt,model=model,n=1,temperature=TEMP)
    print(res)
    fixed_code = list(set([ x['text'].encode()  for x in res['choices'] if "error" not in x.keys() ]))
    return fixed_code, code, work_dir/vul_code[3][2:]
def getMeta(localId):
    work_dir    = tmpDir()
    # Get meta data
    pname       = getPname(localId)
    diff_file   = getDiff(localId)
    vul_commit  = getVulCommit(localId)
    if vul_commit == False or diff_file == False:
        return False
    # Copy Repo
    repo_dir    = work_dir / pname 
    url         = get_projectInfo(localId,pname)[0]['url']
    if _check_repo(OSS_DB/b58encode(url).decode()/pname,localId) == False:
        eventLog(f"[-] XxX: Failed to prepare the main repo: {localId}")
        shutil.rmtree(work_dir)
        return False
    shutil.copytree( OSS_DB/b58encode(url).decode()/pname, repo_dir,symlinks=True)
    
    # Check out to vul version
    if check_call(['git','reset','--hard',vul_commit],repo_dir) == False:
        shutil.rmtree(work_dir)
        return False
    # Get code info, maker sure there should be only one case
    vul_code = get_vul_code(diff_file,repo_dir)
    res = []
    for x in vul_code:
        res.append(x[0])
    # if(len(vul_code)!=1):
    #     print(f"[X] The case is a complex case. Please user python Functions as API")
    #     eventLog(f"[X] More than one modifications. The result could be not precise: {localId=}")
    #     return False
    # else:
    #     vul_code = vul_code[0]
    return res

def BenchMarkAPI(localId,fix):
    work_dir    = tmpDir()
    # Get meta data
    pname       = getPname(localId)
    diff_file   = getDiff(localId)
    vul_commit  = getVulCommit(localId)
    if vul_commit == False or diff_file == False:
        return False
    # Copy Repo
    repo_dir    = work_dir / pname 
    url         = get_projectInfo(localId,pname)[0]['url']
    if _check_repo(OSS_DB/b58encode(url).decode()/pname,localId) == False:
        eventLog(f"[-] XxX: Failed to prepare the main repo: {localId}")
        shutil.rmtree(work_dir)
        return False
    shutil.copytree( OSS_DB/b58encode(url).decode()/pname, repo_dir,symlinks=True)
    
    # Check out to vul version
    if check_call(['git','reset','--hard',vul_commit],repo_dir) == False:
        shutil.rmtree(work_dir)
        return False
    # Get code info, maker sure there should be only one case
    vul_code = get_vul_code(diff_file,repo_dir)
    if(len(vul_code)!=1):
        eventLog(f"[X] More than one modifications. The result could be not precise: {localId=}")
        return False
    else:
        vul_code = vul_code[0]
    target_file = vul_code[3]  
    
    # Perform Fixing
    # try:
    #     fixes, ori_code, target_file = fixer(localId,vul_code,repo_dir)
    # except:
    #     return False
    
    # Try to build and verify all possibe fixes  
         
    perform_fix(fix,vul_code[0],repo_dir/target_file[2:])
    res = verify_FIX(localId,repo_dir,pname)
    if res:
        print("[+] Successful fix: ")
        print(fix)

    shutil.rmtree(work_dir)
    shutil.rmtree(os.path.dirname(diff_file))
    if res:
        print("[+] SUCCESS!")
        return True
    else:
        print("[-] FAIL to FIX.")
        return False
    
def XxX(localId,module,chance=1,lite=False,logDiff=False,tag=""):
    # logDiff False or a string of Path
    # TODO: CLEAN TMP DIR
    if logDiff == True:
        logDiff = tmpDir() 
    if logDiff != False:
        logDiff = logDiff/ f"{module}_{tag}"
        logDiff.mkdir(exist_ok=True)
    if module == "Codex":
        module = "code-davinci-edit-001"
    if module not in ["gpt-3.5-turbo-16k","Starcoder","gpt-3.5-turbo-instruct","code-davinci-edit-001","gpt-3.5-turbo","gpt-4","gpt-4-1106-preview"] and \
        'Wizard' not in module:
        panic(f"[X] Invalid Model {module}")

    work_dir    = tmpDir()

    # Get meta data
    pname       = getPname(localId)
    diff_file   = getDiff(localId)
    vul_commit  = getVulCommit(localId)
    if vul_commit==False or diff_file==False:
        return False
    # Copy Repo
    repo_dir    = work_dir / pname 
    url         = get_projectInfo(localId,pname)[0]['url']
    if _check_repo(OSS_DB/b58encode(url).decode()/pname,localId) == False:
        eventLog(f"[-] XxX: Failed to prepare the main repo: {localId}")
        shutil.rmtree(work_dir)
        return False
    shutil.copytree( OSS_DB/b58encode(url).decode()/pname, repo_dir,symlinks=True)
    
    # Check out to vul version
    if check_call(['git','reset','--hard',vul_commit],repo_dir) == False:
        shutil.rmtree(work_dir)
        return False
    # Get code info, maker sure there should be only one case
    vul_code = get_vul_code(diff_file,repo_dir)
    

    if(len(vul_code)!=1):
        eventLog(f"[X] More than one modifications. The result could be not precise: {localId=}")
        return False
    else:
        vul_code = vul_code[0]

    # Perform Fixing
    if(module in ["Codex","gpt-3.5-turbo-instrct","code-davinci-edit-001"]):
        try:
            fixes, ori_code, target_file = get_Codex_fix(localId,vul_code,repo_dir,model=module,lite=lite,logDiff=logDiff)
        except:
            eventLog(f"[X] Failed perform Codex fixing: {localId=}")
            return False
    elif("Wizard" in module):
        try:
            fixes, ori_code, target_file = get_Wizard_fix(localId,vul_code,repo_dir,model=module,lite=lite)
            if fixes[0]==b'':
                return False
        except:
            eventLog(f"[X] Failed perform Codex fixing: {localId=}")
            return False
    elif module in ["gpt-3.5-turbo","gpt-4","gpt-4-1106-preview","gpt-3.5-turbo-16k"]:
        try:
            fixes, ori_code, target_file = get_GPT_fix(localId,vul_code,repo_dir,model=module,lite=lite,logDiff=logDiff)
        except:
            eventLog(f"[X] Failed perform GPT fixing: {localId=}")
            return False
    elif module == "Starcoder":
        try:
            fixes, ori_code, target_file = get_star_fix(localId,vul_code,repo_dir,model=module,lite=lite,)
        except:
            eventLog(f"[X] Failed perform Starcoder fixing: {localId=}")
            return False
    else:
        panic("UNK Module")
    
    for x in fixes[:chance]:
        print("\n"+x.decode()+"\n")
    # Try to build and verify all possibe fixes        
    for fix in fixes[:chance]:
        perform_fix(fix,ori_code,target_file)
        res = verify_FIX(localId,repo_dir,pname)
        if res:
            print("[+] Successful fix: ")
            print(fix)
            break
    if chance == 1 and logDiff!=False:
        dst= logDiff / str(localId)
        dst.mkdir(exist_ok=True)
        diff_content = genDiff(vul_code[0],fixes[0])
        fixDiff(diff_content,dst,f"fix_{localId}.diff")
        oracleDiff(diff_file,dst,f"ora_{localId}.diff")



    shutil.rmtree(work_dir)
    shutil.rmtree(os.path.dirname(diff_file))
    if logDiff:
        dst= logDiff / str(localId)
        with open(dst/"res",'w') as f:
            if res:
                f.write(f"[+] SUCCESS!\n")
            else:
                f.write(f"[-] FAIL to FIX.\n")
    if res:
        print("[+] SUCCESS!")
        return True
    else:
        print("[-] FAIL to FIX.")
        return False

def patchDesc(localId,update=False):
    if not update and Path(f"./PatchDesc/{localId}.log").exists():
        return 
    
    patch_desc = oss_fuzz_vul_labeler(localId)
    if patch_desc == False:
        eventLog(f"[-] patchDesc: Faild to analysis the pacth: {localId=}")
        return
    with open(f"./PatchDesc/{localId}.log","w") as f:
        f.write(patch_desc)
    print("Sleeping to cool-down...")
    time.sleep(10)
    return

def desc_allPatches(ids=None,update=False):
    if ids != None:
        pass
    else:
        ids = get_all_single_mods()
    for x in ids:
        print(f"[+] Generating the Desc for issue: {x}")
        patchDesc(x,update)
    return 

# Test Area
def TestBenchmark(round, module="GPT"):
    logfile= f"./_fx_local_log/{module}_round_{round}.json"
    logfile = Path(logfile)
    # Init log files
    if not json_file_check(logfile):
        panic(f"[X] Failed to init {logfile}")
    # Get tested issues
    content = json.loads(open(logfile).read())
    tested = [int(x.strip())for x in content.keys()]
    # Do test
    ids = get_all_single_mods(False)
    while(len(tested) < len(ids)):
        chosen = ids[random.randint(0,len(ids)-1)]
        if chosen in tested:
            continue
        try:
            res=LMFix(chosen, module)
        except:
            res=False
        content[chosen] = res
        with open(logfile,'w') as f:
            json.dumps(content,sort_keys = True,indent=4)
    print("[!] Test ALL Done")
def Benchmark(round, module):
    if module not in ['GPT','Codex','Wizard']:
        print("[X] Select a module from 'GPT','Codex','Wizard'")
        return False
    logfile= f"./Log/BenchMark/{module}_round_{round}.json"
    logfile = Path(logfile)
    # Init log files
    if not json_file_check(logfile):
        panic(f"[X] Failed to init {logfile}")
    # Get tested issues
    content = json.loads(open(logfile).read())
    tested = [int(x.strip())for x in content.keys()]
    # Do test
    ids = get_all_single_mods(False)
    while(len(tested) < len(ids)):
        chosen = ids[random.randint(0,len(ids)-1)]
        if chosen in tested:
            continue
        try:
            res= XxX(chosen, module)
        except:
            res=False
        content[chosen] = res
        with open(logfile,'w') as f:
            json.dumps(content,sort_keys = True,indent=4)
    print("[!] Test ALL Done")
### returns prompt
def getWizardPrompt(localId, vul_code,lite=False):
    if lite == True:
        desciption = getCrashType(localId)
    else:
        patch_desc = getDesc(localId)
        desciption = patch_desc['vul']
    prompt = f"""
Instruction:
Rewrite this code to patch the bug:
```
{vul_code}
```
    
Bug description:

{desciption}

Always and only return the rewritten code.
"""
    return prompt
### Generate input.jsonl for HPC Wizard
def GenerateWizardInput(input_file_path, localIds=[58086], limit=1,lite=False,logDiff=False):
    # Do test
    input_file = jsonlines.open(input_file_path, mode='w')
    cnt=0
    for localId in localIds:
        if(1):
            vul_code = [b'']
            # TODO: CLEAN TMP DIR
            work_dir    = tmpDir()
            # Get meta data
            pname       = getPname(localId)
            diff_file   = getDiff(localId)
            vul_commit  = getVulCommit(localId)
            if vul_commit==False or diff_file==False:
                raise ValueError('Incorrect vul commit or diff file.')
            # Copy Repo
            repo_dir    = work_dir / pname 
            url         = get_projectInfo(localId,pname)[0]['url']
            if _check_repo(OSS_DB/b58encode(url).decode()/pname,localId) == False:
                eventLog(f"[-] XxX: Failed to prepare the main repo: {localId}")
                shutil.rmtree(work_dir)
                raise ValueError('Failed to prepare the main repo for the issue')
            shutil.copytree( OSS_DB/b58encode(url).decode()/pname, repo_dir,symlinks=True)
            
            # Check out to vul version
            if check_call(['git','reset','--hard',vul_commit],repo_dir) == False:
                eventLog(f"[-] XxX: Failed to reset {localId}")
                shutil.rmtree(work_dir)
                raise ValueError('Failed to git reset to vul commit')

            # Get code info, maker sure there should be only one case
            vul_code = get_vul_code(diff_file,repo_dir)
            # print("[+] Got vul code")
            # print(vul_code)
            if(len(vul_code)>limit):
                print("[X] More than one modifications. The result could be not precise.")
                vul_code = [b'']
            else:
                vul_code = vul_code[0]
            # print(vul_code)
        # except Exception as e:
            # print("[-] Failed getting vul code:" + str(e))
            # vul_code = [b'']
        if vul_code is not [b'']:
            prompt = getWizardPrompt(localId,vul_code[0].decode(),lite)
            input_data = {"idx":localId, "Instruction":prompt}
            input_file.write(input_data)
            cnt+=1
            if logDiff != False:
                dst= logDiff / str(localId)
                dst.mkdir(exist_ok=True)
                with open(dst/"prompt","w") as f:
                    f.write(prompt)
        if cnt==100:
            break
        
    print(f"[+] Finished writing {str(cnt)} prompts to {input_file_path}")

if __name__ =="__main__":
    pass
    