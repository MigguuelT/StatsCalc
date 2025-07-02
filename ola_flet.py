# app.py
import flet as ft

def main(page: ft.Page):
    page.title = "Meu App Incrível"
    page.vertical_alignment = ft.CrossAxisAlignment.CENTER

    page.add(
        ft.Column(
            [
                ft.Text("Bem-vindo ao meu App!", size=40),
                ft.ElevatedButton("Clique-me!", on_click=lambda e: page.add(ft.Text("Você clicou!")))
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
    )

if __name__ == "__main__":
    ft.app(target=main)