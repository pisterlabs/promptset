import numpy as np
import openai
import time


def collapse_r( response, toks ):
    total_prob = 0.0
    for t in toks:
        if t in response:
            total_prob += response[t]
    return total_prob

def lc( t ):
    return t.lower()

def uc( t ):
    return t.upper()

def mc( t ):
    tmp = t.lower()
    return tmp[0].upper() + t[1:]

def gen_variants( toks ):
    results = []

    variants = [ lc, uc, mc ]

    for t in toks:
        for v in variants:
            results.append( " " + v(t) )

    return results

def logsumexp( log_probs ):
    log_probs = log_probs - np.max(log_probs)
    log_probs = np.exp(log_probs)
    log_probs = log_probs / np.sum( log_probs )
    return log_probs

def extract_probs( lp ):
    lp_keys = list( lp.keys() )
    ps = [ lp[k] for k in lp_keys ]
    ps = logsumexp( np.asarray(ps) )
    vals = [ (lp_keys[ind], ps[ind]) for ind in range(len(lp_keys)) ]

    vals = sorted( vals, key=lambda x: x[1], reverse=True )

    result = {}
    for v in vals:
        result[ v[0] ] = v[1]

    return result

def collapse_probs( tok_sets, response ):
    tr = []
    for tok_set_key in tok_sets.keys():
        toks = tok_sets[tok_set_key]
        full_prob = collapse_r( response[0], toks )
        tr.append( full_prob )
#        print( f";{tok_set_key};{full_prob}", end="" )
#        print( "\t{:.2f}".format(full_prob), end="" )
#    print("\t\t",end="")
    tr = np.asarray( tr )
    tr = tr / np.sum(tr)

    return tr

def print_response( tok_sets, probs ):
    for ind, tok_set_key in enumerate( tok_sets.keys() ):
        print( "\t{:.2f}".format(probs[ind]), end="" )
    print("")

def do_query( prompt, max_tokens=2 ):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=1,
        logprobs=100,
    )

    token_responses = response['choices'][0]['logprobs']['top_logprobs']

    results = []
    for ind in range(len(token_responses)):
        results.append( extract_probs( token_responses[ind] ) )


    return results, response
