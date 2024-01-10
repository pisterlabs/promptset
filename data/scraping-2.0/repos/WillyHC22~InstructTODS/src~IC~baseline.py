import os
import json
import torch
import openai
import argparse
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset
from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import get_label_intent_dict, get_labels



def main(args):
    device = torch.device("cuda:0")
    if "clinc" in args.dataset:
        dataset = load_dataset(args.dataset, "small", split="test")
        dataset = dataset.rename_column('intent', 'label')
    else:
        dataset = load_dataset(args.dataset, split="test")
    if args.debug_mode:
        dataset = dataset[:5]

    LABELS = get_labels(args.label_path)

    if args.do_inference:
        if "clinc" in args.dataset:
            candidate_labels_raw = [label.split(" ")[1] for label in LABELS.split("\n")]
            if args.corr:
                candidate_labels = [label.replace("_", " ") for label in candidate_labels_raw]
            else:
                candidate_labels = [label for label in candidate_labels_raw]
        else:
            intent_labels = """activate_my_card, age_limit, apple_pay_or_google_pay, atm_support, automatic_top_up, balance_not_updated_after_bank_transfer, balance_not_updated_after_cheque_or_cash_deposit, beneficiary_not_allowed, cancel_transfer, card_about_to_expire, card_acceptance, card_arrival, card_delivery_estimate, card_linking, card_not_working, card_payment_fee_charged, card_payment_not_recognised, card_payment_wrong_exchange_rate, card_swallowed, cash_withdrawal_charge, cash_withdrawal_not_recognised, change_pin, compromised_card, contactless_not_working, country_support, declined_card_payment, declined_cash_withdrawal, declined_transfer, direct_debit_payment_not_recognised, disposable_card_limits, edit_personal_details, exchange_charge, exchange_rate, exchange_via_app, extra_charge_on_statement, failed_transfer, fiat_currency_support, get_disposable_virtual_card, get_physical_card, getting_spare_card, getting_virtual_card, lost_or_stolen_card, lost_or_stolen_phone, order_physical_card, passcode_forgotten, pending_card_payment, pending_cash_withdrawal, pending_top_up, pending_transfer, pin_blocked, receiving_money, Refund_not_showing_up, request_refund, reverted_card_payment?, supported_cards_and_currencies, terminate_account, top_up_by_bank_transfer_charge, top_up_by_card_charge, top_up_by_cash_or_cheque, top_up_failed, top_up_limits, top_up_reverted, topping_up_by_card, transaction_charged_twice, transfer_fee_charged, transfer_into_account, transfer_not_received_by_recipient, transfer_timing, unable_to_verify_identity, verify_my_identity, verify_source_of_funds, verify_top_up, virtual_card_not_working, visa_or_mastercard, why_verify_identity, wrong_amount_of_cash_received, wrong_exchange_rate_for_cash_withdrawal"""
            candidate_labels_raw = intent_labels.split(", ")
            candidate_labels = [label.replace("_", " ") for label in candidate_labels_raw]

        pipe = pipeline("zero-shot-classification",
                        model=args.model,
                        device=device)

        outputs = []
        for text in tqdm(dataset["text"]):

            output = pipe(text,
                        candidate_labels=candidate_labels,
            )
            outputs.append(output)
        
        save_path = os.path.join(args.save_dir, args.save_file)
        with open(save_path, "w") as f:
            json.dump(outputs, f, indent=4)

    if args.do_evaluation:

        label2intent, intent2label = get_label_intent_dict(LABELS)

        eval_file_path = os.path.join(args.save_dir, args.eval_file)
        results_json = json.load(open(eval_file_path, "r"))
        golds = [label2intent[str(label)] for label in dataset["label"]]
        if args.corr:
            preds1 = [label["labels"][0].replace(" ", "_") for label in results_json]
            preds2 = [label["labels"][1].replace(" ", "_") for label in results_json]
            preds3 = [label["labels"][2].replace(" ", "_") for label in results_json]
        else:
            preds1 = [label["labels"][0] for label in results_json]
            preds2 = [label["labels"][1] for label in results_json]
            preds3 = [label["labels"][2] for label in results_json]
            
        def compute_prf(golds, preds):
            average = "macro"
            accuracy = accuracy_score(golds, preds)
            precision = precision_score(golds, preds, average=average)
            recall = recall_score(golds, preds, average=average)
            f1 = f1_score(golds, preds, average=average)
            evaluation = {"Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,}
            return evaluation
        
        evaluation = {"top1":compute_prf(golds, preds1),
                      "top2":compute_prf(golds, preds2),
                      "top3":compute_prf(golds, preds3)}

        results_processed = []
        for idx, result in enumerate(results_json):
            results_processed.append({"text":result["sequence"],
                                      "gold":golds[idx],
                                      "pred":{"top1":preds1[idx],
                                              "top2":preds2[idx],
                                              "top3":preds3[idx]}})
            

        save_path_res = eval_file_path[:-5] + "_evalresults.json"
        with open(save_path_res, "w") as f:
            json.dump(results_processed, f, indent=4) 

        save_path = eval_file_path[:-5] + "_evalscore.json"
        with open(save_path, "w") as f:
            json.dump(evaluation, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/bart-large-mnli", help="model to use")
    parser.add_argument("--dataset", type=str, default="banking77", help="dataset for intent classification")
    parser.add_argument("--save_file", type=str, default="test_run.json", help="save file")
    parser.add_argument("--debug_mode", action="store_true", help="use this argument for subset test")
    parser.add_argument("--save_dir", type=str, default="/home/willy/instructod/src/IC/results/baselines/", help="save directory")
    parser.add_argument("--eval_file", type=str, default="test_run.json", help="save directory")
    parser.add_argument("--do_inference", action="store_true", help="use this argument to do inference the output after having done inference")
    parser.add_argument("--do_evaluation", action="store_true", help="use this argument to evaluate the output after having done inference")
    parser.add_argument("--label_path", type=str, default="/home/willy/instructod/src/IC/config/intents.txt", help="txt file with the intent labels")
    parser.add_argument("--as_topn", type=int, default=1, help="get n-th prediction for the classification")
    parser.add_argument("--corr", action="store_true", help="deletes underscore for bart to better understand the intent")
    args = parser.parse_args()
    main(args)