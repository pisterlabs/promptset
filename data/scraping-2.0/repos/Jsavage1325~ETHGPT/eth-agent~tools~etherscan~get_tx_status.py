from langchain.tools import BaseTool
from tools.etherscan.EtherScanBase import EtherScanBase


class EtherScanGetTXStatus(EtherScanBase):
    name = 'get_ether_tx_status'
    description = 'Gets the receipt status for an ethereum transaction using a transaction hash, where possible.'

    def _run(self, txhash: str):
        """
        Gets the etherscan contract code
        """
        payload = {
            "module": "transaction",
            "action": "gettxreceiptstatus",
            "txhash": txhash,
        }
        return self._get(payload)

    def _arun(self, address: str):
        """
        Gets the etherscan contract code
        """
        raise NotImplementedError()
        