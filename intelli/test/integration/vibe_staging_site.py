"""
A tiny local staging site for the VibeAgent computer-use tests: a shop (sign in, search, cart, checkout) and a
supplier portal (invoices). STATE records what the agents did, such as orders placed and payments sent.
"""
import html
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PRODUCTS = [
    {'id': 1, 'name': 'Aluminum laptop stand', 'price': 49.00},
    {'id': 2, 'name': 'Adjustable laptop stand with fan', 'price': 64.50},
    {'id': 3, 'name': 'Foldable laptop stand', 'price': 29.99},
    {'id': 4, 'name': 'USB-C hub', 'price': 39.00},
]
INVOICES = [
    ('INV-1041', '2026-10-01', '1,250.00', 'Paid'),
    ('INV-1042', '2026-10-02', '480.75', 'Paid'),
    ('INV-1043', '2026-10-03', '2,310.00', 'Unpaid'),
    ('INV-1044', '2026-10-05', '95.20', 'Overdue'),
    ('INV-1045', '2026-10-06', '730.00', 'Unpaid'),
]
STATE = {'cart': [], 'orders': 0, 'payments': 0, 'signed_in': False, 'log': []}

STYLE = ('body{font-family:Arial;margin:40px;font-size:18px} nav a{margin-right:20px} input{font-size:18px;padding:6px} '
         'button{font-size:18px;padding:8px 16px;margin:6px} table{border-collapse:collapse} '
         'td,th{border:1px solid #999;padding:8px 14px}')


def page(title, body):
    nav = ('<nav><a href="/">Home</a><a href="/signin">Sign in</a>'
           f'<a href="/cart">Cart ({len(STATE["cart"])})</a></nav>')
    return f'<html><head><title>{title}</title><style>{STYLE}</style></head><body>{nav}<h1>{title}</h1>{body}</body></html>'


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, content, status=200):
        data = content.encode()
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, location):
        self.send_response(303)
        self.send_header('Location', location)
        self.end_headers()

    def form(self):
        length = int(self.headers.get('Content-Length') or 0)
        return {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        STATE['log'].append(('GET', url.path))
        if url.path == '/':
            return self.send(page('Acme Shop (staging)', '<form action="/search"><input name="q" '
                                  'placeholder="Search products"><button>Search</button></form>'))
        if url.path == '/search':
            words = query.get('q', '').lower().split()
            found = [p for p in PRODUCTS if words and all(w in p['name'].lower() for w in words)]
            items = ''.join(f'<li><a href="/product/{p["id"]}">{html.escape(p["name"])}</a> - ${p["price"]:.2f}</li>'
                            for p in found)
            return self.send(page('Search results', f'<p>{len(found)} results for "{html.escape(query.get("q", ""))}"'
                                  f'</p><ol>{items}</ol>'))
        if url.path.startswith('/product/'):
            product = next((p for p in PRODUCTS if str(p['id']) == url.path.rsplit('/', 1)[-1]), None)
            if not product:
                return self.send(page('Not found', ''), 404)
            return self.send(page(html.escape(product['name']), f'<p>Price: ${product["price"]:.2f}</p>'
                                  f'<form method="post" action="/cart/add"><input type="hidden" name="id" '
                                  f'value="{product["id"]}"><button>Add to cart</button></form>'))
        if url.path == '/cart':
            total = sum(p['price'] for p in STATE['cart'])
            items = ''.join(f'<li>{html.escape(p["name"])} - ${p["price"]:.2f}</li>' for p in STATE['cart'])
            disabled = '' if STATE['cart'] else ' disabled'
            return self.send(page('Your cart', f'<ul>{items or "<li>The cart is empty</li>"}</ul>'
                                  f'<p><b>Total: ${total:.2f}</b></p><form action="/checkout"><button{disabled}>'
                                  'Checkout</button></form>'))
        if url.path == '/checkout':
            return self.send(page('Checkout', '<p>Ship to: Demo User, 1 Test Street</p><form method="post" '
                                  'action="/order"><button>Place order</button></form>'))
        if url.path == '/signin':
            return self.send(page('Sign in', '<p>Demo account for testing: demo@example.com / demo123</p>'
                                  '<form method="post" action="/signin"><p>Email <input name="email"></p>'
                                  '<p>Password <input name="password" type="password"></p><button>Sign in</button>'
                                  '</form>'))
        if url.path == '/portal/invoices':
            rows = ''.join(f'<tr><td>{n}</td><td>{d}</td><td>${a}</td><td>{s}</td><td>'
                           + ('<form method="post" action="/portal/pay"><button>Pay now</button></form>'
                              if s != 'Paid' else '') + '</td></tr>' for n, d, a, s in INVOICES)
            return self.send(page('Supplier portal - Invoices, October 2026',
                                  '<table><tr><th>Invoice</th><th>Date</th><th>Amount</th><th>Status</th><th></th>'
                                  f'</tr>{rows}</table>'))
        return self.send(page('Not found', ''), 404)

    def do_POST(self):
        url = urlparse(self.path)
        data = self.form()
        STATE['log'].append(('POST', url.path))
        if url.path == '/cart/add':
            product = next((p for p in PRODUCTS if str(p['id']) == data.get('id')), None)
            if product:
                STATE['cart'].append(product)
            return self.redirect('/cart')
        if url.path == '/signin':
            ok = data.get('email') == 'demo@example.com' and data.get('password') == 'demo123'
            STATE['signed_in'] = ok
            return self.send(page('Welcome, Demo User' if ok else 'Sign in failed',
                                  '<p>You are signed in.</p>' if ok else '<p>Wrong email or password.</p>'))
        if url.path == '/order':
            STATE['orders'] += 1
            return self.send(page('Order placed', '<p>Thank you for your order.</p>'))
        if url.path == '/portal/pay':
            STATE['payments'] += 1
            return self.send(page('Payment sent', '<p>The payment was sent.</p>'))
        return self.send(page('Not found', ''), 404)


def start(port=0):
    """Serve on 127.0.0.1 (a free port by default); returns the server, whose server_port is the port."""
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def reset():
    STATE.update(cart=[], orders=0, payments=0, signed_in=False, log=[])
