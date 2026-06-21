"""
╔═╗╔╦╗╦═╗╔═╗╔╦╗╔═╗═╗ ╦
║═╬╗║ ╠╦╝╠═╣ ║║║╣ ╔╩╦╝
╚═╝╚╩ ╩╚═╩ ╩═╩╝╚═╝╩ ╚═

execution.py


Execution class for interacting with a cryptocurrency exchange via the CCXT library.

Methods:
    __init__(self, exchange_id, api_key=None, api_secret=None):
        Initializes the CCXTWrapper with the provided exchange and API credentials.

    create_order(self, self.symbol, side, order_type, amount, price):
        Creates a limit order on the exchange.

    cancel_order(self, order_id, self.symbol):
        Cancels a single open order.

    cancel_orders(self, order_ids, self.symbol):
        Cancels multiple open orders.

    cancel_all_orders(self, self.symbol):
        Cancels all open orders for a given self.symbol asynchronously.

    fetch_open_order(self, order_id, self.symbol=None):
        Fetches details of a specific open order.

    fetch_open_orders(self, self.symbol=None):
        Fetches all open orders for a given self.symbol.

    fetch_ticker(self, self.symbol, params=None):
        Fetches the current price ticker for a given self.symbol.

    place_market_order_with_depth(self, self.symbol, side, amount, depth_percent):
        Places a market order at a specified depth percentage below or above the current market price.

    drip_by_time(self, self.symbol, side, total_amount, chunks, pause_seconds):
        Places repeat market orders in chunks with a pause between each order.

    drip_to_limit(self, self.symbol, side, total_amount, chunks, pause_seconds, limit_price):
        Places repeat market orders in chunks with a pause, but temporarily stops if the price is below a set limit.

    drip_to_market(self, self.symbol, side, total_amount, chunks):
        Places repeat market orders in chunks, waiting for the market to respond before proceeding with the next order.

    iceberg_order(self, self.symbol, side, total_amount, iceberg_limit):
        Monitors price and places orders to maintain the iceberg limit until the total amount is exhausted.

    top_of_book(self, self.symbol, side, total_amount):
        Places a market order at the top of the order book, continuously monitoring bid/ask prices.
"""


import threading
import time

import ccxt
class Execution:
    def __init__(self, exchange_id, asset, currency, api_key=None, api_secret=None):
        """
        Initialize the CCXTWrapper with the provided exchange and API credentials.
        """
        self.symbol = f"{asset}/{currency}"
        if exchange_id == "bitshares":
            from qtradex.private.bitshares_exchange import BitsharesExchange
            self.exchange = BitsharesExchange(user=api_key, wif=api_secret)
        else:
            self.exchange = getattr(ccxt, exchange_id)(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                }
            )
        self.killswitch = [False]

    def create_order(self, side, order_type, amount, price):
        """
        Create a limit order.
        """
        try:
            order = self.exchange.create_order(self.symbol, order_type, side, amount, price)
            return order
        except Exception as e:
            return str(e)

    def cancel_order(self, order_id):
        """
        Cancel a single open order.
        """
        try:
            canceled_order = self.exchange.cancel_order(order_id, self.symbol)
            return canceled_order
        except Exception as e:
            return str(e)

    def cancel_orders(self, order_ids):
        """
        Cancel multiple open orders.
        """
        try:
            canceled_orders = self.exchange.cancel_orders(order_ids, self.symbol)
            return canceled_orders
        except Exception as e:
            return str(e)

    def cancel_all_orders(self):
        """
        Cancel all open orders for a given self.symbol.
        """
        self.killswitch[0] = True
        try:
            canceled_orders = self.exchange.cancel_all_orders(self.symbol)
            return canceled_orders
        except Exception as e:
            return str(e)

    def cancel_order_threads(self):
        self.killswitch[0] = True

    def fetch_open_order(self, order_id):
        """
        Fetch details of an open order.
        """
        try:
            open_order = self.exchange.fetch_open_order(order_id, self.symbol)
            return open_order
        except Exception as e:
            return str(e)

    def fetch_open_orders(self):
        """
        Fetch all open orders for a given self.symbol.
        """
        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            return open_orders
        except Exception as e:
            return str(e)

    def fetch_my_trades(self):
        """
        Fetch all open orders for a given self.symbol.
        """
        try:
            open_orders = self.exchange.fetch_my_trades(self.symbol)
            return open_orders
        except Exception as e:
            return str(e)

    def fetch_ticker(self, symbol, params=None):
        """
        Fetch the price ticker for a given self.symbol.
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol, params)
            return ticker
        except Exception as e:
            return str(e)

    # Create a market order at a certain depth below the ticker price
    def create_market_order(self, side, amount, depth_percent=50):
        """
        Places a market order at a specified depth percentage below the current market price.

        :param self.symbol: Market self.symbol (e.g., 'BTC/USDT')
        :param side: 'buy' or 'sell'
        :param amount: Amount of base currency to trade
        :param depth_percent: The percentage below the market price to place the order (negative for buy, positive for sell)
        :return: Order details
        """
        self.killswitch = [False]
        ticker = self.exchange.fetch_ticker(self.symbol)
        price = ticker["ask"] if side == "buy" else ticker["bid"]

        adjusted_price = (
            price * (1 - depth_percent / 100)
            if side == "buy"
            else price * (1 + depth_percent / 100)
        )

        try:
            order = self.create_order(self.symbol, side, "limit", amount, adjusted_price)
            return order
        except Exception as e:
            return str(e)

    # Drip by time - places repeat market orders in chunks, with pause in between
    def drip_by_time(self, side, total_amount, chunks, pause_seconds):
        """
        Places repeat market orders in chunks, every so many seconds, until the total amount is exhausted.

        :param self.symbol: Market self.symbol (e.g., 'BTC/USDT')
        :param side: 'buy' or 'sell'
        :param total_amount: Total amount of base currency to trade
        :param chunks: Number of chunks to divide the total amount into
        :param pause_seconds: Pause in seconds between each order
        """
        self.killswitch = [False]
        chunk_size = total_amount / chunks

        def place_orders(self):
            for _ in range(chunks):
                if self.killswitch[0]:
                    break
                try:
                    order = self.create_market_order(self.symbol, side, chunk_size)
                    print(f"Placed order: {order}")
                except Exception as e:
                    print(f"Error placing order: {str(e)}")
                time.sleep(pause_seconds)

        self.drip_thread = threading.Thread(target=place_orders, args=(self,))
        self.drip_thread.start()

    # Drip to limit - places repeat market orders but temporarily pauses if ticker price is below the limit
    def drip_to_limit(
        self, side, total_amount, chunks, pause_seconds, limit_price
    ):
        """
        Places repeat market orders in chunks, with pause in between, until the total amount is exhausted.
        The order pauses if the price is below a certain limit.

        :param self.symbol: Market self.symbol (e.g., 'BTC/USDT')
        :param side: 'buy' or 'sell'
        :param total_amount: Total amount of base currency to trade
        :param chunks: Number of chunks to divide the total amount into
        :param pause_seconds: Pause in seconds between each order
        :param limit_price: The price below which we temporarily pause the dripping
        """
        self.killswitch = [False]
        chunk_size = total_amount / chunks

        def place_orders(self):
            for _ in range(chunks):
                if self.killswitch[0]:
                    break
                ticker = self.exchange.fetch_ticker(self.symbol)
                price = ticker["ask"] if side == "buy" else ticker["bid"]

                if (side == "buy" and price <= limit_price) or (
                    side == "sell" and price >= limit_price
                ):
                    print("Price below limit, pausing drip...")
                    while (side == "buy" and price <= limit_price) or (
                        side == "sell" and price >= limit_price
                    ):
                        time.sleep(pause_seconds)
                        ticker = self.exchange.fetch_ticker(self.symbol)
                        price = ticker["ask"] if side == "buy" else ticker["bid"]

                try:
                    order = self.create_market_order(self.symbol, side, chunk_size)
                    print(f"Placed order: {order}")
                except Exception as e:
                    print(f"Error placing order: {str(e)}")
                time.sleep(pause_seconds)

        self.drip_thread = threading.Thread(target=place_orders, args=(self,))
        self.drip_thread.start()

    # Drip to market - places repeat market orders but waits for the market to respond (next buy/sell)
    def drip_to_market(self, side, total_amount, chunks):
        """
        Places repeat market orders in chunks, with pause in between, and waits for the market to respond.

        :param self.symbol: Market self.symbol (e.g., 'BTC/USDT')
        :param side: 'buy' or 'sell'
        :param total_amount: Total amount of base currency to trade
        :param chunks: Number of chunks to divide the total amount into
        """
        self.killswitch = [False]
        chunk_size = total_amount / chunks

        def place_orders(self):
            for _ in range(chunks):
                if self.killswitch[0]:
                    break
                # Wait for the market to respond (next trade or price change)
                while True:
                    ticker = self.exchange.fetch_ticker(self.symbol)
                    price = ticker["ask"] if side == "buy" else ticker["bid"]
                    if side == "buy" and price < ticker["ask"]:
                        break
                    if side == "sell" and price > ticker["bid"]:
                        break
                    time.sleep(5)

                try:
                    order = self.create_market_order(self.symbol, side, chunk_size)
                    print(f"Placed order: {order}")
                except Exception as e:
                    print(f"Error placing order: {str(e)}")
                time.sleep(5)

        self.drip_thread = threading.Thread(target=place_orders, args=(self,))
        self.drip_thread.start()

    # Iceberg order - monitors price and places buy/sell to maintain iceberg limit
    def iceberg_order(self, side, total_amount, iceberg_limit):
        """
        Monitors price and places buy/sell orders to maintain the iceberg limit until the amount is exhausted.

        :param self.symbol: Market self.symbol (e.g., 'BTC/USDT')
        :param side: 'buy' or 'sell'
        :param total_amount: Total amount of base currency to trade
        :param iceberg_limit: The iceberg limit price for buying/selling
        """
        self.killswitch = [False]
        amount_remaining = total_amount
        order_size = total_amount / 10  # Divide total amount into 10 parts for iceberg

        def place_orders():
            while amount_remaining > 0 and not self.killswitch[0]:
                ticker = self.exchange.fetch_ticker(self.symbol)
                price = ticker["ask"] if side == "buy" else ticker["bid"]

                if (side == "buy" and price <= iceberg_limit) or (
                    side == "sell" and price >= iceberg_limit
                ):
                    try:
                        order = self.create_market_order(self.symbol, side, order_size)
                        print(f"Placed iceberg order: {order}")
                        amount_remaining -= order_size
                    except Exception as e:
                        print(f"Error placing order: {str(e)}")
                time.sleep(5)

        self.iceberg_thread = threading.Thread(target=place_orders, args=(self,))
        self.iceberg_thread.start()

    # Top of the book - keeps full order at the top of the book
    def top_of_book(self, side, total_amount):
        """
        Monitors the bid/ask and keeps the full order at the top of the book.

        :param self.symbol: Market self.symbol (e.g., 'BTC/USDT')
        :param side: 'buy' or 'sell'
        :param total_amount: Total amount of base currency to trade
        """
        self.killswitch = [False]

        def place_order(self):
            while not self.killswitch[0]:
                ticker = self.exchange.fetch_ticker(self.symbol)
                price = ticker["ask"] if side == "buy" else ticker["bid"]
                try:
                    order = self.create_market_order(self.symbol, side, total_amount)
                    print(f"Placed order at top of book: {order}")
                except Exception as e:
                    print(f"Error placing order: {str(e)}")
                time.sleep(5)

        self.top_of_book_thread = threading.Thread(target=place_order, args=(self,))
        self.top_of_book_thread.start()
