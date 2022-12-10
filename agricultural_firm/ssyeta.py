import http
aa = ""
class handler_(http.server.SimpleHTTPRequestHandler):
    def do_CONNECT(self):
        print(self.path)
        print(self.client_address)
        print(self.headers)
        print(len(self.headers))
        return super().handle_one_request()
server = http.server.HTTPServer(("localhost", 8080), handler_)
server.serve_forever()