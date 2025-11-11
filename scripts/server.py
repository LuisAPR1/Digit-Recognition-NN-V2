#!/usr/bin/env python3
"""
Servidor HTTP simples para servir os ficheiros do projeto
Resolve o problema de CORS ao abrir HTML diretamente
"""

import http.server
import socketserver
import webbrowser
import os

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Adicionar headers CORS para permitir carregar ficheiros
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def main():
    # Mudar para o diretório raiz do projeto (um nível acima do script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        url = f"http://localhost:{PORT}/web/index.html"
        print(f"🚀 Servidor iniciado em http://localhost:{PORT}")
        print(f"📂 A servir ficheiros de: {os.getcwd()}")
        print(f"🌐 Abrindo {url} no navegador...")
        print(f"\n⚠️  Para parar o servidor, pressiona Ctrl+C\n")
        
        # Abrir automaticamente no navegador
        try:
            webbrowser.open(url)
        except:
            print(f"⚠️  Não foi possível abrir o navegador automaticamente.")
            print(f"   Por favor, abre manualmente: {url}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n🛑 Servidor parado.")

if __name__ == "__main__":
    main()

