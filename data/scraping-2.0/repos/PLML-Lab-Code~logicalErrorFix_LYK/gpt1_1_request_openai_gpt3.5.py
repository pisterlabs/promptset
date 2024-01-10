# How to Use:
# OPENAI_API_KEY=<YOUR_KEY> python lyk/gpt1_1_request_openai_gpt3.5.py

import os

import backoff
import openai
from ratelimiter import RateLimiter
from tqdm import tqdm, trange

from utils import gpt1_json_prompt_reader

# 분당 최대 3500번의 호출을 수행할 수 있는 rate_limiter를 생성 (gpt-3.5-turbo 제한 분당 3,500요청 /90,000 토큰)
rate_limiter = RateLimiter(max_calls=3200, period=60)

# [program_id]\t[correct_id]\t[incorrect_id]\t[correct_code]\t[error_code]\t[error_code_stmt]
DATASET = 'data/edit_distance/refined_pair_code_edit_dist_valid.txt'
OUTPUT_DIR = 'lyk/output/gpt1_lshv2fs/'
MODEL = 'gpt-3.5-turbo'
#PROMPT_SYSTEM_ZERO_SHOT = "You are a CPP error solver. As a request, you are given a code file delimited by [line number][space][code]. In your response, print the line of code where the error occurred on the first line, and fix that line of code on the second line. If you don't have any code to fix, simply print -1 on the first line and an empty space on the second line. You shouldn't give any response other than the two lines."
# PROMPT_SYSTEM_ZERO_SHOT = "당신은 cpp 오류 해결사입니다. 요청으로 로지컬 오류가 포함되어 있을 수 있는 코드 파일이 주어집니다. ||| 사이에 있는 코드를 수정시켜주세요. 여러개의 오류가 있는 경우 첫번째 오류 라인과 코드만 출력해야 합니다. 출력은 다음 형식을 따라주세요: \n에러 줄: [에러가_발생한_줄]\t[수정된_코드]\n\n"
# 당신은 cpp 오류 해결사입니다. 요청으로 [줄번호][공백][코드] 로 줄줄이 구분된 코드 파일이 주어집니다.
# 응답의 첫번째 줄에는 에러가 발생한 코드의 줄을 출력하고 두번째 줄에는 해당 라인의 코드를 고쳐서 출력하세요.
# # 고칠 코드가 없는 경우에는 첫번째 줄에 -1, 두번째 줄에는 빈칸을 출력하면 됩니다.
# 2개의 줄을 제외한 다른 응답을 해선 안됩니다.

#PROMPT_SYSTEM_ZERO_SHOT = "code의 ||| 사이에 있는 코드 부분 중 한개에 수정이 필요할때 수정된 부분만 stmt에 적어서 stmt만 출력해줘"

# PROMPT = '''code의 ||| 사이에 있는 코드 부분 중 한개에 수정이 필요할때 수정된 부분만 stmt에 적어서 stmt만 출력해줘

# {{"code": "{}", "line_no": "", "stmt": ""}}
# '''

# PROMPT = '''code의 ||| 사이에 있는 코드 부분 중 한개에 수정이 필요할때 수정된 부분만 stmt, line_no에 적어서 stmt, line_no만 출력해줘

# ```{{"code": "{}", "line_no": "", "stmt": ""}}```
# '''

# PROMPT = '''code가 ||| 로 줄 구분된 오류가 있는 코드이며, 수정된 부분의 줄번호를 line_no에 적고 수정한 내용을 stmt에 적어서 출력해줘.

# ```{{"code": "{}", "line_no": "", "stmt": ""}}```
# '''

# PROMPT = '''code has errors separated lines by |||, write the corrected code in stmt and the line number of the corrected part in line_no and print it out.
# PROMPT = '''code has errors separated lines by |||, write the corrected code in stmt and the line number of the corrected part in line_no and print it out.

# ```{{"code": "{}", "line_no": "", "stmt": ""}}```
# '''

# PROMPT = '''code is separated by |||, find the logicial error and fill the json form below.

#{}'''

# PROMPT = '''find the logicial error and fill the json form below.
# do not write any additional information.

# {}

# {{"line_number_with_error": "", "original_stmt": "", "fixed_stmt": ""}}'''

#13	388	212	"1 #include <bits/stdc++.h>||| 2 using namespace std;||| 3 signed main() {||| 4 int t;||| 5 cin >> t;||| 6 while (t--) {||| 7 string s;||| 8 cin >> s;||| 9 int n = s.size();||| 10 int cnt = 0;||| 11 for (int i = 0; i < n; i++) {||| 12 cnt += s[i] == 'B';||| 13 }||| 14 if (cnt + cnt == n)||| 15 cout << "YES\n";||| 16 else||| 17 cout << "NO\n";||| 18 }||| 19 }||| "	"1 #include <bits/stdc++.h>||| 2 using namespace std;||| 3 signed main() {||| 4 int t;||| 5 cin >> t;||| 6 while (t--) {||| 7 string s;||| 8 cin >> s;||| 9 int n = s.size();||| 10 int cnt = 0;||| 11 for (int i = 0; i < n; i++) {||| 12 cnt += s[i] == 'B';||| 13 }||| 14 if (cnt + cnt >= n)||| 15 cout << "YES\n";||| 16 else||| 17 cout << "NO\n";||| 18 }||| 19 }||| "	14 if (cnt + cnt == n)
# 1 #include <bits/stdc++.h>||| 2 using namespace std;||| 3 signed main() {||| 4 int t;||| 5 cin >> t;||| 6 while (t--) {||| 7 string s;||| 8 cin >> s;||| 9 int n = s.size();||| 10 int cnt = 0;||| 11 for (int i = 0; i < n; i++) {||| 12 cnt += s[i] == 'B';||| 13 }||| 14 if (cnt + cnt >= n)||| 15 cout << "YES\n";||| 16 else||| 17 cout << "NO\n";||| 18 }||| 19 }||| 
# 14
# if (cnt + cnt >= n)
# if (cnt + cnt == n)
# 28	42	288	"1 #include <bits/stdc++.h>||| 2 using namespace std;||| 3 void solve() {||| 4 int m, n, o, x, y, z, p, q, c = 0, l;||| 5 scanf("%d %d %d", &m, &n, &o);||| 6 if ((m + o) % 2 == 0) {||| 7 printf("0\n");||| 8 } else {||| 9 printf("1\n");||| 10 }||| 11 }||| 12 int main() {||| 13 int t;||| 14 scanf("%d", &t);||| 15 for (int i = 1; i <= t; i++) {||| 16 solve();||| 17 }||| 18 return 0;||| 19 }||| "	"1 #include <bits/stdc++.h>||| 2 using namespace std;||| 3 void solve() {||| 4 int m, n, o, x, y, z, p, q, c = 0, l;||| 5 scanf("%d %d %d", &m, &n, &o);||| 6 if ((m + o) % 2 == 0) {||| 7 printf("1\n");||| 8 } else {||| 9 printf("0\n");||| 10 }||| 11 }||| 12 int main() {||| 13 int t;||| 14 scanf("%d", &t);||| 15 for (int i = 1; i <= t; i++) {||| 16 solve();||| 17 }||| 18 return 0;||| 19 }||| "	"7 printf("0\n");"
# 1 #include <bits/stdc++.h>||| 2 using namespace std;||| 3 void solve() {||| 4 int m, n, o, x, y, z, p, q, c = 0, l;||| 5 scanf("%d %d %d", &m, &n, &o);||| 6 if ((m + o) % 2 == 0) {||| 7 printf("1\n");||| 8 } else {||| 9 printf("0\n");||| 10 }||| 11 }||| 12 int main() {||| 13 int t;||| 14 scanf("%d", &t);||| 15 for (int i = 1; i <= t; i++) {||| 16 solve();||| 17 }||| 18 return 0;||| 19 }||| 
# 7
# 틀: printf("1\n");
# 맞: printf("0\n");

PROMPT = '''find the logicial error and fill the json form below.
do not write any additional information.

{}

{{"line_number_with_error": "", "original_stmt": "", "fixed_stmt": ""}}'''


def create_prompt(code):
  # If code is coverd in front and back with " or ', remove them
  if code[0] == '"' and code[-1] == '"':
    code = code[1:-1]
  if code[0] == "'" and code[-1] == "'":
    code = code[1:-1]
  # Replace " with \"
  code = code.replace('"', '\\"')
  # return PROMPT.format(code)
  return code

@backoff.on_exception(backoff.expo, Exception, max_tries=16) # openai.error.RateLimitError
def openai_chat_cpp_error_solver(prompt):
  # print("debug: prompt: ", prompt)
  # print("debug: reader: ", gpt1_json_prompt_reader('lyk/gpt1/prompt/lshv2.json', prompt))
  # openai.Completion.create
  response = openai.ChatCompletion.create(
    model=MODEL,
    messages=gpt1_json_prompt_reader('lyk/gpt1/prompt/lshv2fs.json', prompt),
    # [
    #   # {"role": "system", "content": PROMPT_SYSTEM_ZERO_SHOT},
    #   # {"role": "user", "content":  '1 #include <bits/stdc++.h>||| 2 using namespace std;||| 3 int main() {||| 4 int t;||| 5 cin >> t;||| 6 while (t--) {||| 7 int c = 0;||| 8 string s;||| 9 cin >> s;||| 10 for (int i = 0; i < s.length(); i++) {||| 11 if (s[i] == \'N\') c++;||| 12 }||| 13 if (c > 1)||| 14 cout << ""YES"" << endl;||| 15 else||| 16 cout << ""NO"" << endl;||| 17 }||| 18 return 0;||| 19 }||| '},
    #   # {"role": "assistant", "content": '13\nif (c != 1)'},
    #   {"role": "user", "content": prompt},
    # ],
    temperature=0,
  )
# <OpenAIObject chat.completion id=chatcmpl-6xpmlDodtW6RwiaMaC1zhLsR8Y1D3 at 0x10dccc900> JSON: {
#   "choices": [
#     {
#       "finish_reason": "stop", # or "length" if max_tokens is reached
#       "index": 0,
#       "message": {
#         "content": "Orange who?",
#         "role": "assistant"
#       }
#     }
#   ],
#   "created": 1679718435,
#   "id": "chatcmpl-6xpmlDodtW6RwiaMaC1zhLsR8Y1D3",
#   "model": "gpt-3.5-turbo-0301",
#   "object": "chat.completion",
#   "usage": {
#     "completion_tokens": 3,
#     "prompt_tokens": 39,
#     "total_tokens": 42
#   }
# }
  return response['choices'][0]['message']['content']

#sample = '''1 #include <bits/stdc++.h>||| 2 using namespace std;||| 3 const long long int mod = 998244353;||| 4 const long long int inf = 1e18;||| 5 const long long int N = 5000;||| 6 long long int factorialNumInverse[N + 1];||| 7 long long int naturalNumInverse[N + 1];||| 8 long long int fact[N + 1];||| 9 void InverseofNumber(long long int p) {||| 10 naturalNumInverse[0] = naturalNumInverse[1] = 1;||| 11 for (long long int i = 2; i <= N; i++)||| 12 naturalNumInverse[i] = naturalNumInverse[p % i] * (p - p / i) % p;||| 13 }||| 14 void InverseofFactorial(long long int p) {||| 15 factorialNumInverse[0] = factorialNumInverse[1] = 1;||| 16 for (long long int i = 2; i <= N; i++)||| 17 factorialNumInverse[i] =||| 18 (naturalNumInverse[i] * factorialNumInverse[i - 1]) % p;||| 19 }||| 20 void factorial(long long int p) {||| 21 fact[0] = 1;||| 22 for (long long int i = 1; i <= N; i++) {||| 23 fact[i] = (fact[i - 1] * i) % p;||| 24 }||| 25 }||| 26 signed main() {||| 27 ios_base::sync_with_stdio(false);||| 28 cin.tie(NULL);||| 29 cout.tie(NULL);||| 30 long long int p = mod;||| 31 InverseofNumber(p);||| 32 InverseofFactorial(p);||| 33 factorial(p);||| 34 long long int n, k;||| 35 string s;||| 36 cin >> n >> k >> s;||| 37 long long int l = 0, r = 0;||| 38 vector<long long int> a;||| 39 for (long long int i = 0; i < n; i++) {||| 40 if (s[i] == '1') a.push_back(i);||| 41 }||| 42 if (a.size() < k || k == 0) {||| 43 cout << 1 << ""\n"";||| 44 return 0;||| 45 }||| 46 r = (a.size() == k ? n - 1 : a[k] - 1);||| 47 long long int ans =||| 48 (fact[r - l + 1] *||| 49 (factorialNumInverse[k] * factorialNumInverse[r - l + 1 - k]) % mod) %||| 50 mod;||| 51 long long int pl = l, pr = r;||| 52 bool done = 0;||| 53 long long int g = k;||| 54 while (g < a.size()) {||| 55 pr = r;||| 56 pl = l;||| 57 l = a[g - k] + 1;||| 58 r = (g + 1 == a.size() ? n - 1 : a[g + 1] - 1);||| 59 long long int t1 =||| 60 (fact[r - l + 1] *||| 61 (factorialNumInverse[k] * factorialNumInverse[r - l + 1 - k] % mod)) %||| 62 mod;||| 63 long long int x = pr - l + 1;||| 64 long long int t2 = (fact[x] * (factorialNumInverse[k - 1] *||| 65 factorialNumInverse[x - k + 1] % mod)) %||| 66 mod;||| 67 ans = (ans + t1 - t2 + mod) % mod;||| 68 g++;||| 69 }||| 70 cout << ans << ""\n"";||| 71 return 0;||| 72 }||| '''
#sample = '''1 #include <bits/stdc++.h>||| 2 using namespace std;||| 3 void solve() {||| 4 int n, m, rb, cb, rd, cd;||| 5 scanf(""%d%d%d%d%d%d"", &m, &n, &rb, &cb, &rd, &cd);||| 6 int ans = 0;||| 7 int dr = 1, dc = 1;||| 8 while (rb != rd && cb != cd) {||| 9 if (rb + dr < 1 || rb + dr > n) dr *= -1;||| 10 if (cb + dc < 1 || cb + dc > m) dc *= -1;||| 11 rb += dr;||| 12 cb += dc;||| 13 ans++;||| 14 }||| 15 printf(""%d\n"", ans);||| 16 }||| 17 int main() {||| 18 int t;||| 19 scanf(""%d"", &t);||| 20 while (t--) {||| 21 solve();||| 22 }||| 23 }||| '''
#sample = '''1 #include <bits/stdc++.h>||| 2 using namespace std;||| 3 const long long int mod = 998244353;||| 4 const long long int inf = 1e18;||| 5 const long long int N = 5000;||| 6 long long int factorialNumInverse[N + 1];||| 7 long long int naturalNumInverse[N + 1];||| 8 long long int fact[N + 1];||| 9 void InverseofNumber(long long int p) {||| 10 naturalNumInverse[0] = naturalNumInverse[1] = 1;||| 11 for (long long int i = 2; i <= N; i++)||| 12 naturalNumInverse[i] = naturalNumInverse[p % i] * (p - p / i) % p;||| 13 }||| 14 void InverseofFactorial(long long int p) {||| 15 factorialNumInverse[0] = factorialNumInverse[1] = 1;||| 16 for (long long int i = 2; i <= N; i++)||| 17 factorialNumInverse[i] =||| 18 (naturalNumInverse[i] * factorialNumInverse[i - 1]) % p;||| 19 }||| 20 void factorial(long long int p) {||| 21 fact[0] = 1;||| 22 for (long long int i = 1; i <= N; i++) {||| 23 fact[i] = (fact[i - 1] * i) % p;||| 24 }||| 25 }||| 26 signed main() {||| 27 ios_base::sync_with_stdio(false);||| 28 cin.tie(NULL);||| 29 cout.tie(NULL);||| 30 long long int p = mod;||| 31 InverseofNumber(p);||| 32 InverseofFactorial(p);||| 33 factorial(p);||| 34 long long int n, k;||| 35 string s;||| 36 cin >> n >> k >> s;||| 37 long long int l = 0, r = 0;||| 38 vector<long long int> a;||| 39 for (long long int i = 0; i < n; i++) {||| 40 if (s[i] == '1') a.push_back(i);||| 41 }||| 42 if (a.size() < k || k == 0) {||| 43 cout << 1 << ""\n"";||| 44 return 0;||| 45 }||| 46 r = (a.size() == k ? n - 1 : a[k] - 1);||| 47 long long int ans =||| 48 (fact[r - l + 1] *||| 49 (factorialNumInverse[k] * factorialNumInverse[r - l + 1 - k]) % mod) %||| 50 mod;||| 51 long long int pl = l, pr = r;||| 52 bool done = 0;||| 53 long long int g = k;||| 54 while (g < a.size()) {||| 55 pr = r;||| 56 pl = l;||| 57 l = a[g - k] + 1;||| 58 r = (g + 1 == a.size() ? n - 1 : a[g + 1] - 1);||| 59 long long int t1 =||| 60 (fact[r - l + 1] *||| 61 (factorialNumInverse[k] * factorialNumInverse[r - l + 1 - k] % mod)) %||| 62 mod;||| 63 long long int x = pr - l + 1;||| 64 long long int t2 = (fact[x] * (factorialNumInverse[k - 1] *||| 65 factorialNumInverse[x - k + 1] % mod)) %||| 66 mod;||| 67 ans = (ans + t1 - t2 + mod) % mod;||| 68 g++;||| 69 }||| 70 cout << ans << ""\n"";||| 71 return 0;||| 72 }||| '''
def main():
  # clear terminal via printing \n 20 times
  print('\n' * 20)
  # Show information
  print(f'Using dataset: {DATASET}, output dir: {OUTPUT_DIR}, model: {MODEL}')

  # create dir
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  # open dataset
  with open(DATASET, 'r') as f:
    # tqdm: progress bar
    lines = f.readlines()
    for line in tqdm(lines):
      # get all data
      data = line.strip().split('\t')
      program_id = data[0]
      correct_id = data[1]
      incorrect_id = data[2]
      # correct_code_raw = data[3]
      incorrect_code_raw = data[4]
      # error_code_stmt = data[5]

      # skip header
      if (program_id == 'PID'):
        continue
      
      # skip if file already exists
      file_path = os.path.join(OUTPUT_DIR, f'P{program_id}C{correct_id}I{incorrect_id}.txt')
      if os.path.exists(file_path):
        continue

      incorrect_code_prompt = create_prompt(incorrect_code_raw)
      response = openai_chat_cpp_error_solver(incorrect_code_prompt)
      # Write to file P[program_id]C[correct_id]I[incorrect_id].txt
      with open(file_path, 'w') as f1:
        f1.write(response)

if __name__ == '__main__':
  main()
