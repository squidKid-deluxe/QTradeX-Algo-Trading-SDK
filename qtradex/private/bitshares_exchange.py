import bitshares_signing.rpc as bitshares_rpc
from bitshares_signing import broker, prototype_order
from bitshares_signing.config import NODES

DEV = True


class BitsharesExchange:
    def __init__(self, user, wif):
        self.rpc = bitshares_rpc.wss_handshake()
        self.account_name = user
        self.account_id = bitshares_rpc.rpc_get_account(self.rpc, self.account_name)[
            "id"
        ]
        if DEV:
            print(f"Account name: {self.account_name}\nAccount ID: {self.account_id}")
        self.wif = wif

        self.login()

    def _prototype(self, symbol):
        asset, currency = symbol.split("/")
        try:
            info = {
                "asset_id": bitshares_rpc.id_from_name(self.rpc, asset),
                "currency_id": bitshares_rpc.id_from_name(self.rpc, currency),
                "account_id": self.account_id,
                "account_name": self.account_name,
                "wif": self.wif,
            }
            info.update(
                {
                    "asset_precision": bitshares_rpc.precision(
                        self.rpc, info["asset_id"]
                    ),
                    "currency_precision": bitshares_rpc.precision(
                        self.rpc, info["currency_id"]
                    ),
                }
            )
            order = prototype_order(info)
        except RuntimeError:
            self.rpc = bitshares_rpc.wss_handshake()
            return self._prototype(symbol)
        return order

    def login(self):
        try:
            rpc_name = bitshares_rpc.rpc_get_objects(self.rpc, self.account_id)["name"]
        except RuntimeError:
            self.rpc = bitshares_rpc.wss_handshake()
            return self.login()

        if self.account_name != rpc_name:
            raise ValueError(
                f'Invalid account name! "{self.account_name}" does not exist on BitShares.'
            )
        order = {
            "edicts": [{"op": "login"}],
            "header": {
                "account_id": self.account_id,
                "account_name": self.account_name,
                "wif": self.wif,
            },
            "nodes": NODES,
        }
        if not broker(order) and not DEV:
            raise ValueError("Failed to authenticate!")

    def create_order(self, symbol, order_type, side, amount, price):
        order = self._prototype(symbol)

        if order_type == "swap":
            pass
            # FIXME implement swaps
        if order_type == "limit":
            if side not in ["buy", "sell"]:
                raise ValueError
            order["edicts"].append({"op": side, "amount": amount, "price": price})
            return broker(order)
            # FIXME order id callback
        else:
            raise ValueError(f"Invalid order_type {order_type}")

    def cancel_order(self, order_id, symbol):
        order = self._prototype(symbol)
        order["edicts"].append({"op": "cancel", "ids": [order_id]})
        return broker(order)

    def cancel_orders(self, ids, symbol):
        order = self._prototype(symbol)
        order["edicts"].append({"op": "cancel", "ids": ids})
        return broker(order)

    def cancel_all_orders(self, symbol):
        order = self._prototype(symbol)
        order["edicts"].append({"op": "cancel", "ids": ["1.7.X"]})
        return broker(order)

    def fetch_open_order(self, order_id, _):
        try:
            return bitshares_rpc.rpc_get_objects(self.rpc, order_id)
        except RuntimeError:
            self.rpc = bitshares_rpc.wss_handshake()
            return self.fetch_open_order(order_id, None)

    def fetch_open_orders(self, symbol):
        pair = self._prototype(symbol)["header"]
        return bitshares_rpc.rpc_open_orders(self.rpc, self.account_name, pair)

    def fetch_balance(self):
        try:
            return {"free": bitshares_rpc.rpc_balances(self.rpc, self.account_name)}
        except RuntimeError:
            self.rpc = bitshares_rpc.wss_handshake()
            return self.fetch_balance()

    def fetch_ticker(self, symbol):
        """
        Returns:
            {
                "bid": float(),
                "ask": float(),
            }
        """
        pair = self._prototype(symbol)["header"]
        return bitshares_rpc.rpc_ticker(self.rpc, pair["asset_id"], pair["currency_id"])
