"""import http
aa = ""
class handler_(http.server.SimpleHTTPRequestHandler):
    def do_CONNECT(self):
        print(self.path)
        print(self.client_address)
        print(self.headers)
        print(len(self.headers))
        return super().handle_one_request()
server = http.server.HTTPServer(("localhost", 8080), handler_)
server.serve_forever()"""
c = 2
def a(a=1, b=2):
    print(a,b)
    print(c)

arg = {"a":1,"b":2,"c":23}

from datetime import datetime, timedelta

print((-12 - 1) // 13)
print(abs(-12 + 1) // 13)