import pyautogui


class CursorController:

    def __init__(self):

        # se elimina la pausa automatica que agrega pyautogui
        pyautogui.PAUSE = 0

        # se obtiene el tamaño de la pantalla
        self.screen_width, self.screen_height = pyautogui.size()


    def move(self, normalized_x, normalized_y):

        # se convierte la posicion del dedo a coordenadas de pantalla
        mouse_x = int(
            normalized_x * self.screen_width
        )

        mouse_y = int(
            normalized_y * self.screen_height
        )


        # se evita que el cursor salga de los limites de la pantalla
        mouse_x = max(
            0,
            min(
                self.screen_width - 1,
                mouse_x
            )
        )

        mouse_y = max(
            0,
            min(
                self.screen_height - 1,
                mouse_y
            )
        )


        # se mueve el mouse a la nueva posicion
        pyautogui.moveTo(
            mouse_x,
            mouse_y,
            duration=0
        )