import streamlit as st
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random 
import sympy
from sympy import *
import struct
import pandas as pd 
import plotly.express as px 
import plotly.graph_objects as go
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import struct
import sympy as sp 
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import binascii

transformations = (standard_transformations +
                   (implicit_multiplication_application,))

st.set_page_config(page_title="Calculadoras",
                   page_icon=":pencil:", layout="wide")
st.sidebar.image(".\images\mmu.png", use_column_width=True)
menu = ["Biseccion",
        "Simpson 1/3",
        "Expresiones",
        "Derivadas",
        "Cambio de Bases",
        "Falsa Posicion",
        "Metodo de la Secante",
        "Newton Rapson", "Graficadora",
        "IEEE",
        "Simpson 3/8", "Raices de un polinomio", "Montecarlo",
        "Trapecios", "Integrales izq-der-med","Matrices", "evaluador de matrices" , "minimos cuadrados"]

choice = st.sidebar.selectbox("Seleccione una opción", menu)


if choice == "Simpson 1/3":

    def evaluacion(x):
        copia = list(funcion)
        for j in range(len(copia)):
            if copia[j] == "x":
                copia[j] = str(x)
        return eval("".join(copia))

    def simps_method(funcion, a, b, n):
        h = (b - a) / n
        total = 0

        for i in range(1, n):
            x = a + (i * h)
            if (i % 2 == 0):
                total += 2 * evaluacion(x)
            else:
                total += 4 * evaluacion(x)

        total += evaluacion(a) + evaluacion(b)
        total = total * ((1 / 3) * h)

        return total

    st.title("Método de Integración Numérica: Simpson 1/3")
    st.write("Ingrese los datos para la integral/función:")

    funcion = st.text_input("Función")
    a = st.number_input("Intervalo inferior", value=0.0 ,step=0.0000000000001, format="%f")
    b = st.number_input("Intervalo superior", value=0.0, step=0.0000000000001, format="%f")
    n = st.number_input("Valor de n", value=1, step=1)

    if st.button("Calcular"):
        resultado = simps_method(funcion, a, b, n)
        st.write(f"Resultado de la aproximación: {resultado}")

        # Definimos un rango de valores para x
        x = np.linspace(a, b, num=100)
        # Evaluamos la función en cada punto del rango de valores de x
        y = [evaluacion(i) for i in x]

        # Creamos la tabla
        tabla = []
        tabla.append(['Subintervalo', 'Puntos evaluados',
                     'f(x)', 'Aproximación de la integral'])
        h = (b - a) / n
        for i in range(n):
            xi = a + (i * h)
            xf = a + ((i + 1) * h)
            puntos = np.linspace(xi, xf, num=2)
            fx = [evaluacion(j) for j in puntos]
            aproximacion = simps_method(funcion, xi, xf, 2)
            tabla.append([f'{i+1}', f'{puntos}', f'{fx}', f'{aproximacion}'])

        st.table(tabla)

        # Creamos la gráfica
        plt.plot(x, y)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Gráfica de la función')
        # Incluimos la función en formato LaTeX en la etiqueta de la gráfica
        plt.text((a+b)/2, max(y), f'$f(x)={funcion}$', ha='center', va='top')
        st.pyplot(plt)

elif choice == "Raices de un polinomio":
    def encontrar_raices(coeficientes_str):
        if coeficientes_str.strip().endswith(","):
            coeficientes_str = coeficientes_str.strip()[:-1]

        coeficientes = [float(x) for x in coeficientes_str.split(",")]
        soluciones = np.roots(coeficientes)
        contador = 1
        soluciones_str = []

        for solucion in soluciones:
            if solucion.imag == 0:
                solucion_str = str(round(solucion.real, 4))
            else:
                solucion_str = str(round(solucion.real, 4)) + " + " + str(round(solucion.imag, 4)) + "i"

            solucion_str = solucion_str.replace("j", "i")
            soluciones_str.append("x{} = {}".format(contador, solucion_str))
            contador += 1

        return soluciones_str
    def coeficientes_a_latex(coeficientes):
        latex_str = ""
        for i, coef in enumerate(coeficientes):
            exponente = len(coeficientes) - i - 1
            if coef == 0:
                continue
            elif coef > 0 and i > 0:
                latex_str += " + "
            elif coef < 0:
                latex_str += " - "
                coef = abs(coef)

            if exponente == 0:
                latex_str += str(coef)
            elif exponente == 1:
                if coef != 1:
                    latex_str += "{}x".format(coef)
                else:
                    latex_str += "x"
            else:
                if coef != 1:
                    latex_str += "{}x^{{{}}}".format(coef, exponente)  # Corrección aquí
                else:
                    latex_str += "x^{{{}}}".format(exponente)  # Corrección aquí

        return latex_str.replace("\n", "")

    def graficar_polinomio(coeficientes_str):
        if coeficientes_str.strip().endswith(","):
            coeficientes_str = coeficientes_str.strip()[:-1]

        coeficientes = [float(x) for x in coeficientes_str.split(",")]
        x = np.linspace(-10, 10, 1000)
        y = np.polyval(coeficientes, x)
        fig, ax = plt.subplots()

        ax.plot(x, y)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')

        return fig
    st.title("Calculadora de Raíces de Polinomios")

    coeficientes_str = st.text_input("Ingrese los coeficientes del polinomio separados por comas (ejemplo: 1,-2,3):")

    if coeficientes_str:
        try:
            coeficientes = [float(x) for x in coeficientes_str.split(",")]

            st.header("Polinomio ingresado:")
            latex_polinomio = coeficientes_a_latex(coeficientes)
            st.latex(latex_polinomio)

            soluciones_str = encontrar_raices(coeficientes_str)

            st.header("Raíces del polinomio:")
            raices_df = pd.DataFrame(soluciones_str, columns=["Raíces"])
            st.table(raices_df)

            st.header("Graficando el polinomio...")
            fig = graficar_polinomio(coeficientes_str)
            st.pyplot(fig)

        except ValueError:
            st.error("Por favor, ingrese coeficientes válidos separados por comas")
           

elif choice == "Expresiones":

    import math
    import streamlit as st

    funciones = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "asinh": math.asinh,
        "acosh": math.acosh,
        "atanh": math.atanh,
        "ln": math.log,
        "log": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
    }

    def evaluar_expresion(expresion):
        try:
            resultado = eval(expresion, funciones)
            st.write(f"El resultado es: {resultado}")
        except Exception as e:
            st.write("Error:", e)

    st.title("Calculadora de expresiones matemáticas")
    expresion = st.text_input("Ingresa una expresión matemática:")

    if st.button("Calcular"):
        if expresion:
            evaluar_expresion(expresion)

    st.write("Expresiones matemáticas disponibles:")
    for funcion in funciones:
        st.write(funcion)

    


elif choice == "Derivadas":
    def calcular_derivadas(funcion, variable):
        # Convertir la ecuación en una expresión SymPy
        x = sympy.symbols(variable)
        f = sympy.sympify(funcion)

        # Calcular la primera derivada
        df = sympy.diff(f, x)

        # Calcular la segunda derivada
        ddf = sympy.diff(df, x)

        # Devolver las derivadas como expresiones SymPy
        respuesta = [df, ddf]

        return respuesta

    # Configurar la página de Streamlit
    st.title("Calculadora de derivadas")
    expresion = st.text_input("Ingresa una función:")
    variable = st.text_input("Ingresa la variable de la función:", "x")

    # Calcular las derivadas cuando el usuario hace clic en el botón
    if st.button("Calcular derivadas"):
        df, ddf = calcular_derivadas(expresion, variable)
        st.write("Primera derivada:", df)
        st.write("Segunda derivada:", ddf)

elif choice == "Cambio de Bases":
    
    import streamlit as st
    import struct

    def float_to_bin(num, precision):
        if precision == '32':
            packed = struct.pack('!f', num)
            integers = struct.unpack('!I', packed)[0]
            binary = format(integers, '032b')
        elif precision == '64':
            packed = struct.pack('!d', num)
            integers = struct.unpack('!Q', packed)[0]
            binary = format(integers, '064b')
        return binary

    def binary_to_hex(binary):
        hex_value = hex(int(binary, 2))[2:].zfill(len(binary) // 4)
        return hex_value

    def binary_to_decimal(binary, precision):
        if precision == '32':
            if binary.startswith('1'):
                sign_bit = -1
            else:
                sign_bit = 1
            exponent = int(binary[1:9], 2) - 127
            significand = int(binary[9:], 2) / (2 ** 23)
            decimal = sign_bit * (1 + significand) * (2 ** exponent)
        elif precision == '64':
            if binary.startswith('1'):
                sign_bit = -1
            else:
                sign_bit = 1
            exponent = int(binary[1:12], 2) - 1023
            significand = int(binary[12:], 2) / (2 ** 52)
            decimal = sign_bit * (1 + significand) * (2 ** exponent)
        return decimal

    def main():
        st.title("Conversor de Punto Flotante IEEE 754")

        precision = st.selectbox("Selecciona la precisión", ['32', '64'])
        num = st.number_input("Ingresa un número de punto flotante" , step=0.0000000000001, format="%f")

        if st.button("Convertir"):
            binary = float_to_bin(num, precision)
            signo = binary[0]
            exponente = binary[1:9] if precision == '32' else binary[1:12]
            significado = binary[9:] if precision == '32' else binary[12:]

            st.subheader("Representación Binaria (IEEE 754):")
            st.text(binary)

            st.subheader("Tabla de Valores:")
            tabla = {
                'Signo': [signo],
                'Exponente': [exponente],
                'Significado': [f"1.{significado}"]
            }
            st.table(tabla)

            hex_value = binary_to_hex(binary)
            decimal_value = binary_to_decimal(binary, precision)

            st.subheader("Valor Hexadecimal:")
            st.text(hex_value)

            st.subheader("Valor Decimal:")
            st.text(f"{decimal_value:.2e}")

    if __name__ == "__main__":
        main()



elif choice == "Falsa Posicion":

    def Falsa_p(funcion, xa, xb, itera=100, error_r=0.001):
        sol = None
        cont = 0
        error_C = 101
        iteraciones = []
        sol_anterior = xa

        if funcion(xa) * funcion(xb) <= 0:
            # calcula la solucion
            while cont <= itera and error_C >= error_r:
                cont += 1
                sol = xb - ((funcion(xb) * (xb - xa)) /
                            (funcion(xb) - funcion(xa)))
                error_C = abs((sol - sol_anterior) / sol) * 100

                # guarda la informacion de la iteracion actual
                iteracion_actual = {
                    'iteracion': cont,
                    'xa': xa,
                    'xb': xb,
                    'sol': sol,
                    'error_C': error_C
                }
                iteraciones.append(iteracion_actual)

                if funcion(xa) * funcion(sol) >= 0:
                    xa = sol
                else:
                    xb = sol

                sol_anterior = sol

            raiz = str('{:.11f}'.format(sol))
            error_calculado = str('{:.3f}'.format(error_C) + '%')

            respuestas1 = [raiz, error_calculado]

            return respuestas1, iteraciones
        else:
            print('no existe solución en ese intervalo')

    st.title("Calculadora de Falsa Posición")

    funcion = st.text_input(
        "Ingrese la función a evaluar", "sin(x)+2*x-3*x/cos(x)")
    xa = st.number_input("Ingrese el valor de a", -10.0, 10.0, -10.0)
    xb = st.number_input("Ingrese el valor de b", -10.0, 10.0, 10.0)
    itera = st.number_input(
        "Ingrese el número máximo de iteraciones", 1, 10000, 100)
    error_r = st.number_input("Ingrese el error máximo", 0.0000000000001, 1.0, 0.001 , step=0.0000000000001, format="%f")

    if st.button("Calcular"):
        respuestas, iteraciones = Falsa_p(
            lambda x: eval(funcion), xa, xb, itera, error_r)
        st.write(f"La raíz encontrada es: {respuestas[0]}")
        st.write(f"El error relativo es: {respuestas[1]}")

        # Convertimos las iteraciones en un DataFrame de pandas para mostrarlo en una tabla
        df_iteraciones = pd.DataFrame(iteraciones)

        # Graficamos la evolución del error relativo
        fig, ax = plt.subplots()
        ax.plot(df_iteraciones["iteracion"], df_iteraciones["error_C"])
        ax.set_xlabel("Iteración")
        ax.set_ylabel("Error relativo (%)")
        st.pyplot(fig)

        # Mostramos las iteraciones en una tabla
        st.write("Iteraciones:")
        st.table(df_iteraciones[["iteracion", "xa", "xb", "sol", "error_C"]])

elif choice == "Biseccion":
    # se hace la funcion 
    def Biseccion(funcion, x_i, x_f, iteraciones=1000, error_rel=0.001):
        # inicializa variables
        solucion = None
        contador = 0
        error_calculado = 101
        iteraciones_guardadas = []
        # evaluar si la raiz esta dentro del intervalo
        if funcion(x_i) * funcion(x_f) <= 0:
            # calcula la solucion
            while contador <= iteraciones and error_calculado >= error_rel:
                contador += 1
                solucion = (x_i + x_f) / 2
                error_calculado = abs((solucion - x_i)/solucion)*100
                # definir el nuevo intervalo
                if funcion(x_i) * funcion(solucion) >= 0:
                    x_i = solucion
                else:
                    x_f = solucion
                # guardar la informacion de la iteracion actual
                iteracion_actual = {
                    'iteracion': contador,
                    'x_i': x_i,
                    'x_f': x_f,
                    'sol': solucion,
                    'error_C': '{:.15f}'.format(error_calculado) + '%' 
                }
                iteraciones_guardadas.append(iteracion_actual)

            if contador <= iteraciones:
                raiz = str('{:.10f}'.format(solucion))
                error_calculado = str('{:.10f}'.format(error_calculado) + '%')
                total_iteraciones = str('{:.0f}'.format(contador))

                respuestas = [raiz, error_calculado,total_iteraciones ]

            else:
                sin_solucion = str('No se pudo encontrar una solucion en las iteraciones dadas')
                respuestas = [sin_solucion, sin_solucion, sin_solucion]
                print('No se pudo encontrar una solucion en las iteraciones dadas')

            return respuestas, iteraciones_guardadas

        else:
            sin_solucion = str('No existe Solucion en ese intervalo')
            respuestas = [sin_solucion, sin_solucion, sin_solucion]
            print('No existe Solucion en ese intervalo')

            return respuestas, iteraciones_guardadas

    st.title("Calculadora de Biseccion ")

    funcion_user = st.text_input(
        "Ingrese la función a evaluar", "x^3 - 2*x^2 - 5")
    
    x_i = st.number_input("Ingrese el valor inicial", value=0.0)

    x_f = st.number_input("Ingrese el valor final", value=0.0)
    
    error_r = st.number_input("Ingrese el error de tolerancia ET", value=0.0, step=0.0000000000001, format="%f")
    iteraciones= 1000
    
    # Convertir la ecuación en una expresión sympy
    x = sympy.symbols('x')
    Funcion_equacioon = sympy.sympify(funcion_user)
    # Convertir la expresión sympy en una función que se puede evaluar
    funcion = sympy.lambdify(x, Funcion_equacioon)
    
    if st.button("Calcular"):
        respuestas, iteraciones = Biseccion(funcion, x_i, x_f, iteraciones, error_r)
        
        st.write(f"La raíz encontrada es: {respuestas[0]}")
        st.write(f"El error relativo es: {respuestas[1]}")
        st.write(f"Numero de iteraciones es: {respuestas[2]}")

        # Convertimos las iteraciones en un DataFrame de pandas para mostrarlo en una tabla
        df_iteraciones = pd.DataFrame(iteraciones)

        # Graficamos la evolución del error relativo
        # fig, x_i = plt.subplots()
        # x_i.plot(df_iteraciones["iteracion"], df_iteraciones["error_C"])
        # x_i.set_xlabel("x")
        # x_i.set_ylabel("y")
        # st.pyplot(fig)

        ##########3

        # Definimos la función a graficar
        def graficar_funcion(funcion, x_i, x_f, solucion):

            # Convertir la ecuación en una expresión sympy
            x = sympy.symbols('x')
            Funcion_ecuacion = sympy.sympify(funcion)
            # Convertir la expresión sympy en una función que se puede evaluar
            funcion_eval = sympy.lambdify(x, Funcion_ecuacion)

            # Crear un arreglo de valores de x para graficar
            x_valores = np.linspace(x_i-5, x_f+5, 200)
            # Evaluar la función para cada valor de x
            y_valores = funcion_eval(x_valores)

            # Graficar la función utilizando Matplotlib
            fig, ax = plt.subplots()
            ax.plot(x_valores, y_valores)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(r'$f(x) = {}$'.format(sympy.latex(Funcion_ecuacion))) # Imprime la función en LaTex
            
            ax.plot(solucion, 0, 'ro')  # Dibuja un punto rojo en la posición de la solución
            ax.annotate(f'Solución: {solucion:.2f}', (solucion, 0), textcoords="offset points", xytext=(-15,10), ha='center', fontsize=8, color='red') # Añade una etiqueta con la solución

            
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            st.pyplot(fig)

        # Entradas del usuario
        # funcion_user = st.text_input("Ingrese la función a evaluar", "x^3 - 2*x^2 - 5")
        # x_i = st.number_input("Ingrese el valor inicial", value=0.0)
        # x_f = st.number_input("Ingrese el valor final", value=0.0)
        # = st.number_input("Ingrese el error de tolerancia ET", value=0.0)
        # iteraciones= 1000
        
        # Llamada a la función para graficar la función ingresada por el usuario
        solucion = float(respuestas[0])
        graficar_funcion(funcion_user, x_i, x_f, solucion)
        st.write("Iteraciones:")
        st.table(df_iteraciones[["iteracion", "x_i", "x_f", "sol", "error_C"]])

elif choice == "Metodo de la Secante":
    def calcular_secante(x0, x1, n, f, ndigits):
        def f_obj(x):
            return eval(f)

        tabla = PrettyTable()
        tabla.field_names = ["Iteración", "Xn-1", "Xn", "Xn+1", "F(Xn+1)", "Error"]

        x_data = []
        y_data = []

        for i in range(n):
            try:
                x2 = x1 - f_obj(x1) * (x1 - x0) / (f_obj(x1) - f_obj(x0))
            except ZeroDivisionError:
                st.warning("Se ha producido una división por cero en la iteración {}. La ejecución ha sido detenida.".format(i+1))
                return

            error = abs(x2 - x1)

            tabla.add_row([i+1, round(x0, ndigits), round(x1, ndigits), round(x2, ndigits), round(f_obj(x2), ndigits), round(error, ndigits)])

            x_data.append(x2)
            y_data.append(f_obj(x2))

            x0 = x1
            x1 = x2

        st.write(tabla)
        st.pyplot(generar_grafico(f, x_data, y_data))

    def generar_grafico(f, x_data, y_data):
        fig, ax = plt.subplots()
        ax.plot(x_data, y_data)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title("Gráfico de la función objetivo")
        return fig

    st.title("Calculadora del Método de la Secante")

    x0 = st.number_input("Valor Inicial 1")
    x1 = st.number_input("Valor Inicial 2")
    n = st.number_input("Número de iteraciones", step=1, min_value=1, max_value=100000)
    f = st.text_input("Función Objetivo")
    ndigits = st.number_input("Número de intervalos después de la coma", step=1, min_value=0, max_value=1000)

    if st.button("Calcular"):
        calcular_secante(x0, x1, n, f, ndigits)
elif choice == "Newton Rapson":

    def eval_function(fun_text, xi):
        """
        Evalúa una función matemática en un punto dado.

        Parameters
        ----------
        fun_text : str
            La función como una cadena de texto.
        xi : float
            El punto en el que se evalúa la función.

        Returns
        -------
        float
            El resultado de evaluar la función en el punto dado.
        """
        x = sympy.Symbol('x')
        fun = sympy.sympify(fun_text)
        return float(fun.subs(x, xi))

    def newton(fun_text, x_n, epsilon, steps):
        """
        Calcula la raíz de una función utilizando el método de Newton-Raphson.

        Parameters
        ----------
        fun_text : str
            La función como una cadena de texto.
        x_n : float
            El punto inicial de la iteración.
        epsilon : float
            La tolerancia del método.
        steps : int
            El número máximo de iteraciones permitidas.

        Returns
        -------
        list of dict
            Una lista de diccionarios, cada uno con información sobre una iteración del método.
        """
        x = sympy.Symbol('x')
        fun = sympy.sympify(fun_text)
        fder = sympy.diff(fun, x)
        results = []
        i = 1
        while i <= steps:
            f_xn = eval_function(fun_text, x_n)
            fder_xn = eval_function(str(fder), x_n)
            x_n1 = x_n - f_xn / fder_xn
            abs_error = abs(x_n1 - x_n)
            rel_error = abs(abs_error / x_n1)
            results.append({
                "iteration": i,
                "approx_root": x_n1,
                "F(xi)": f_xn,
                "f'(xi)": fder_xn,
                "absolute_error": abs_error,
                "relative_error": rel_error
            })
            if abs_error < epsilon:
                break
            x_n = x_n1
            i += 1
        return results

    def main():
        st.title("Método de Newton-Raphson")
        fun_text = st.text_input("Ingrese la función:")
        x_n = st.number_input(
            "Ingrese el punto inicial de la iteración:", value=0.0)
        epsilon = st.number_input(
            "Ingrese la tolerancia del método:", value=1e-6)
        steps = st.number_input(
            "Ingrese el número máximo de iteraciones permitidas:", value=50, step=1)
        if st.button("Calcular"):
            results = newton(fun_text, x_n, epsilon, steps)
            if results:
                st.write(f"La raíz aproximada es {results[-1]['approx_root']}")

    main()

elif choice == "Graficadora":

    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st

    st.title("Graficador de Funciones")

    def reemplaza_funciones(funcion):
        funciones_matematicas = {
            "sin": "np.sin",
            "cos": "np.cos",
            "tan": "np.tan",
            "sqrt": "np.sqrt",
            "exp": "np.exp",
            "log": "np.log",
            "pi": "np.pi",
            "arcsin": "np.arcsin",
            "arccos": "np.arccos",
            "arctan": "np.arctan"
        }
        for f, npf in funciones_matematicas.items():
            funcion = funcion.replace(f, npf)
        return funcion

    # Widgets para ingresar las funciones y los límites de la variable independiente
    num_graficas = st.number_input("Número de gráficas:", value=1, min_value=1, max_value=10, step=1, key="num_graficas")
    funciones = []
    for i in range(num_graficas):
        funcion = st.text_input(f"Ingrese la función {i+1}:", value="sin(x)", key=f"funcion_{i}")
        funciones.append(funcion)
    limite_inferior = st.number_input("Ingrese el límite inferior:", value=-5.0, step=0.1)
    limite_superior = st.number_input("Ingrese el límite superior:", value=5.0, step=0.1)

    # Botón para calcular y graficar las funciones
    if st.button("Calcular"):
        fig, ax = plt.subplots()
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Gráficas de las funciones")

        x = np.linspace(limite_inferior, limite_superior, 1000)

        for i, funcion in enumerate(funciones):
            y = eval(reemplaza_funciones(funcion))
            ax.plot(x, y, label=f"Función {i+1}")

        ax.legend()
        st.pyplot(fig)


elif choice == "Raices de un polinomio":

    def encontrar_raices(coeficientes_str):
        # Verifica si la cadena termina con una coma y la elimina si es necesario
        if coeficientes_str.strip().endswith(","):
            coeficientes_str = coeficientes_str.strip()[:-1]

        # Convierte los coeficientes ingresados a una lista de flotantes
        coeficientes = [float(x) for x in coeficientes_str.split(",")]

        # Encuentra las raíces del polinomio
        soluciones = np.roots(coeficientes)

        # Inicializa un contador para las soluciones
        contador = 1

        # Crea una lista para almacenar las soluciones
        soluciones_str = []

        # Itera sobre cada solución
        for solucion in soluciones:
            # Verifica si la parte imaginaria es cero
            if solucion.imag == 0:
                # Convierte la solución real a una cadena
                solucion_str = str(round(solucion.real, 4))
            else:
                # Convierte la solución compleja a una cadena
                solucion_str = str(round(solucion.real, 4)) + " + " + str(round(solucion.imag, 4)) + "i"

            # Reemplaza "j" por "i" en la cadena
            solucion_str = solucion_str.replace("j", "i")

            # Agrega la solución actualizada a la lista de soluciones
            soluciones_str.append("x{} = {}".format(contador, solucion_str))

            # Incrementa el contador
            contador += 1

    # Retorna la lista de soluciones
            return soluciones_str


    def graficar_polinomio(coeficientes_str):
        # Verifica si la cadena termina con una coma y la elimina si es necesario
        if coeficientes_str.strip().endswith(","):
            coeficientes_str = coeficientes_str.strip()[:-1]

        # Convierte los coeficientes ingresados a una lista de flotantes
        coeficientes = [float(x) for x in coeficientes_str.split(",")]

        # Crea un rango de valores para x
        x = np.linspace(-10, 10, 1000)

        # Evalúa el polinomio para cada valor de x
        y = np.polyval(coeficientes, x)

        # Crea una figura y un eje
        fig, ax = plt.subplots()

        # Grafica la curva del polinomio
        ax.plot(x, y)

        # Configura las etiquetas de los ejes
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')


        # Muestra la figura
        plt.show()


        # Configura la página de Streamlit
        st.set_page_config(page_title="Calculadora de Polinomios", page_icon="📈", layout="wide")

        # Título
        st.title("Calculadora de Polinomios")

        # Instrucciones
        st.write("Ingresa los coeficientes del polinomio separados por comas. Por ejemplo: 1, -3, 2")

        # Entrada de coeficientes
        coeficientes_str = st.text_input("Coeficientes del polinomio:")

        if coeficientes_str:
            # Encuentra las raíces y grafica el polinomio
            raices = encontrar_raices(coeficientes_str)
            st.write("Raíces del polinomio:")
            for raiz in raices:
                st.write(raiz)

            # Graficar el polinomio
            st.write("Gráfico del polinomio:")
            fig = graficar_polinomio(coeficientes_str)

            # Muestra la figura en Streamlit
            st.pyplot(fig)


# elif choice == "IEEE":
elif choice == "Simpson 3/8":

    def f(x):
         # Define la función a integrar
        return x ** 2
        
    def simpson38(f, a, b, n):
        h = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = f(x)
        I = 3 * h / 8 * (y[0] + 3 * y[1:3:n-1].sum() + 2 * y[2:3:n-2].sum() + 3 * y[3:3:n].sum() + y[-1])
        return I

    # Configura la app de Streamlit
    st.title("Calculadora de Simpson 3/8")

    a = st.number_input("Introduce el límite inferior a:", value=0.0, step=0.1)
    b = st.number_input("Introduce el límite superior b:", value=1.0, step=0.1)
    n = st.number_input("Introduce el número de subintervalos (debe ser múltiplo de 3):", value=6, step=1, min_value=3, max_value=1000)
    f_str = st.text_input("Introduce la función a integrar:", "np.sin(x)")

    try:
        f = eval(f_str)  # convierte la cadena en una función
    except:
        st.error("Error al evaluar la función. Por favor verifica la sintaxis e inténtalo de nuevo.")
        st.stop()

    if n % 3 != 0:
        st.warning("El número de subintervalos debe ser múltiplo de 3. Se ajustará automáticamente.")
        n = 3 * (n // 3)

    I = simpson38(f, a, b, n)

    st.write("El resultado de la integración es:", I)

    # Genera una tabla con los valores de la función
    x_vals = np.linspace(a, b, 100)
    y_vals = f(x_vals)
    data = np.column_stack((x_vals, y_vals))
    st.write("Tabla de valores de la función:")
    st.write(data[:10,:], max_rows=10)


    # Genera una gráfica de la función
    fig = plt.figure(figsize=(8,6))  # ajusta el tamaño de la figura
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x_vals, y_vals)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Gráfica de la función")
    st.write(fig)


    # Agrega un botón para calcular de nuevo
    if st.button("Calcular de nuevo"):
        a = 0.0
        b = 1.0
        n = 6
        f_str = "np.sin(x)"
        try:
            f = eval(f_str)  # convierte la cadena en una función
        except:
            st.error("Error al evaluar la función. Por favor verifica la sintaxis e inténtalo de nuevo.")
            st.stop()
        I = simpson38(f, a, b, n)
        st.write("El resultado de la integración es:", I)
# elif choice == "IEEE":
elif choice == "Montecarlo":
    
    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from sympy.parsing.sympy_parser import parse_expr
    from scipy.optimize import minimize_scalar
    import streamlit as st

    # Calcular la cota superior M
    def calcular_cota_superior(ecuacion, a, b):
        x = sp.Symbol('x')
        f = sp.lambdify(x, ecuacion, 'numpy')
        resultado = minimize_scalar(lambda x: -f(x), bounds=(a, b), method='bounded')
        return -resultado.fun

    # Método de Montecarlo
    def montecarlo_integration(ecuacion, a, b, M, num_puntos):
        x = sp.Symbol('x')
        f = sp.lambdify(x, ecuacion, 'numpy')

        puntos_dentro = 0
        area_total = (b - a) * M

        np.random.seed(42)  # Para resultados reproducibles
        for _ in range(num_puntos):
            x_rand = np.random.uniform(a, b)
            y_rand = np.random.uniform(0, M)

            if y_rand <= f(x_rand):
                puntos_dentro += 1

        area_debajo = area_total * (puntos_dentro / num_puntos)
        return area_debajo

    def graficar_funcion(funcion, x_i, x_f, solucion):
        x = sp.Symbol('x')
        funcion_eval = sp.lambdify(x, funcion)

        x_valores = np.linspace(x_i-5, x_f+5, 200)
        y_valores = funcion_eval(x_valores)

        fig, ax = plt.subplots()
        ax.plot(x_valores, y_valores)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r'$f(x) = {}$'.format(sp.latex(funcion)))

        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        st.pyplot(fig)

    # Título
    st.title("Calculadora de Método de Montecarlo")

    # Parámetros de entrada
    st.header("Parámetros de entrada")
    ecuacion_str = st.text_input("Ingrese la ecuación en términos de x:", "x**2")
    a = st.number_input("Límite inferior de integración (a):", value=0.0)
    b = st.number_input("Límite superior de integración (b):", value=1.0)
    num_puntos = st.number_input("Número de puntos:", value=1000, min_value=1, step=1)

    opcion_M = st.radio("¿Desea ingresar la cota superior M manualmente?", ("No", "Sí"))

    if opcion_M == "Sí":
        M = st.number_input("Cota superior de la función en el intervalo (M):", value=1.0)
    else:
        x = sp.Symbol('x')
        ecuacion = parse_expr(ecuacion_str)
        M = calcular_cota_superior(ecuacion, a, b)

    # Resultado
    st.header("Resultado")
    if st.button("Calcular integral"):
        x = sp.Symbol('x')
        ecuacion = parse_expr(ecuacion_str)
        resultado = montecarlo_integration(ecuacion, a, b, M, num_puntos)
        st.write(f"El resultado aproximado de la integral es: {resultado:.6f}")

    # Graficar la función
    graficar_funcion(ecuacion, a, b, resultado)


    # elif choice == "IEEE":
elif choice == "Trapecios":
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt

    # Importar todas las funciones de numpy al espacio de nombres local
    locals().update({k: v for k, v in vars(np).items() if not k.startswith('_')})

    st.title("Calculadora de Integración por Regla del Trapecio")

    fx = st.text_input("Ingrese la función a integrar:")
    a = st.number_input("Ingrese el límite inferior de integración:", step=0.1)
    b = st.number_input("Ingrese el límite superior de integración:", step=0.1)
    tramos = st.number_input("Ingrese el número de tramos:", step=1, format="%d")

    def f(x):
        return eval(fx)

    def regla_del_trapecio(f, a, b, n):
        h = (b-a)/n
        xi = a
        suma = f(xi)
        for i in range(0,n-1,1):
            xi = xi + h
            suma = suma + 2*f(xi)
        suma = suma + f(b)
        integral = h*(suma/2)
        return integral

    if st.button("Calcular"):
        integral = regla_del_trapecio(f, a, b, tramos)
        xip = linspace(a,b,1000)
        fpp = abs(gradient(gradient(f(xip),xip),xip))
        max_fpp = max(fpp)
        error = (-1/12) * ((b-a)/tramos)**3 * max_fpp
    else:
        integral = None
        error = None

    st.write("### Resultados")
    col1, col2 =st.columns(2)
    with col1:
        st.write("#### Integral")
        if integral is not None:
            st.write(integral)
        else:
            st.write("-")
    with col2:
        st.write("#### Error")
        if error is not None:
            st.write(error)
        else:
            st.write("-")

    if integral is not None:
        muestras = tramos + 1
        xi = linspace(a,b,muestras)
        fi = f(xi)
        muestraslinea = tramos*10 + 1
        xk = linspace(a,b,muestraslinea)
        fk = f(xk)

        fig, ax = plt.subplots()
        ax.plot(xk,fk, label ='f(x)')
        ax.plot(xi,fi, marker='o',
                color='orange', label ='muestras')

        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Integral: Regla de Trapecios')
        ax.legend()

        ax.fill_between(xi,0,fi, color='g')
        for i in range(0,muestras,1):
            ax.axvline(xi[i], color='w')

        st.write("### Gráfica")
        st.pyplot(fig)

elif choice == "Integrales izq-der-med":

    def integra_f(funcion, a, b, n):
        h = (b - a) / n
        izq, der, medio = 0, 0, 0
        for i in range(n):
            x_i = a + i * h
            izq += funcion(x_i) * h
            der += funcion(x_i + h) * h
            medio += funcion(x_i + 0.5*h) * h
        izq = str(izq)    
        der = str(der)
        medio = str(medio)
        SolucionIntegracion = [izq, der, medio]
        return SolucionIntegracion

    def mostrar_grafica(funcion, a, b, Funcion_equacioon):
        x = np.linspace(a-5, b+5, 1000)
        y = np.nan_to_num([funcion(xi).evalf().as_real_imag()[0] for xi in x])

        funecuacion = sympy.latex(Funcion_equacioon)

        fig, ax = plt.subplots()
        ax.plot(x, y, label='f(x)')
        ax.axhline(0, color='k', lw=0.6)
        ax.axvline(0, color='k', lw=0.6)
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(r'$%s$' % funecuacion)
        return fig


    st.title("Integración punto izquierdo, derecho y medio")

    a = st.number_input("Introduce el extremo izquierdo a: ", step=0.001)
    b = st.number_input("Introduce el extremo derecho b: ", step=0.001)
    n = st.number_input("Introduce el número de particiones N: ", step=1, min_value=1, format="%i")
    funcion_text = st.text_input("Introduce la función: ")

    if st.button("Calcular"):
        x = sympy.symbols('x')
        Funcion_equacioon = sympy.sympify(funcion_text)
        funcion = sympy.lambdify(x, Funcion_equacioon, 'sympy')

        SolucionIntegracion = integra_f(funcion, a, b, n)
        st.write("Punto extremo izquierdo respuesta: ", SolucionIntegracion[0])
        st.write("Punto extremo derecho respuesta: ", SolucionIntegracion[1])
        st.write("Punto medio respuesta: ", SolucionIntegracion[2])

        fig = mostrar_grafica(funcion, a, b, Funcion_equacioon)
        st.pyplot(fig)
        
elif choice == 'Matrices':
    st.title("Calculadora de Matrices")
    def parsearFuncion(f):
        return parse_expr(f, transformations=transformations)

    def mtr_nm(n,m):
        return np.array([[x for x in range(n)] for y in range(m)])

    def tex_mtr(mtr_str='',sym='A'):
        mtr = sp.Matrix(parse_expr(mtr_str))
        return str(f'{sym} \;= '+sp.latex(mtr))

    def equacion(expr,evaluar=False) -> st.latex:
        return st.latex(sp.latex(expr)) \
            if not evaluar else st.latex(sp.latex(parse_expr(expr)))

    def crear_matriz(sym : str ='A'):
        st.markdown('---')
        _, filas, columnas, _ = st.columns([0.5, 3, 3, 0.5])

        st.markdown('Ingrese matriz:')
        st.latex(sym)
        with filas:
            n_filas = int(st.text_input(
                f'Ingrese el número de filas de la matriz {sym}', '3'))

        with columnas:
            n_columnas = int(st.text_input(
                f'Ingrese el número de columnas de la matriz {sym}', '3'))

        columnas_mtr = st.columns(n_columnas)

        mtr_a = [[(i,j,sym) for j in range(n_columnas)] for i in range(n_filas)]

        idx = 0
        idy = 0
        for col in columnas_mtr:
            with col:
                for _ in range(n_filas):
                    mtr_a[idy][idx] = parsearFuncion(
                        st.text_input('', '0', key=mtr_a[idy][idx]))
                    idy += 1
            idy = 0
            idx += 1
        
        return mtr_a

    def def_mtr(n=1,m=0):
        lineal_o_no = True

        opcion = st.selectbox('Seleccione una opción', 
        [
            'Sumar matrices',
            'Restar matrices',
            'Multiplicar matrices',
            'Producto Hadamard',
            'Determinante e Inversa',
            'Espacio Columna y Nulo',
            'Sistemas de Ecuaciones Lineales',
        
        ])

        operaciones = [
            lambda x,y: x+y, 
            lambda x,y: x-y, 
            lambda x,y: x*y
            ]
        
        if opcion == 'Sistemas de Ecuaciones Lineales':
            lineal_o_no = st.checkbox('¿Es un sistema lineal?', 
                value=True,help='Si el sistema no es lineal, se ingresa distinto.')

        if not lineal_o_no:
            num_e,vars_s = st.columns(2)
            with num_e:
                n_ecuaciones = st.number_input('Ingrese el número de ecuaciones',
                min_value=0,max_value=8,value=2)
            with vars_s:  
                variables = sp.symbols(st.text_input('Ingrese las variables', 'x,y',
                help='Separadas por coma, las variables en el sistema.'))
            ecuaciones = [0 for i in range(n_ecuaciones)]
            ejemplo_sistema = ['xy-1', ' 4*x*2 + y*2 - 5']

            for i in range(n_ecuaciones):
                ecuaciones[i] = parsearFuncion(st.text_input(f'Ingrese la ecuación {i+1}',
                value=ejemplo_sistema[i] if n_ecuaciones == 2 else ecuaciones[i]))
            
            st.markdown('---')
            st.markdown('Solución:')
            st.latex(sp.latex(sp.Matrix(variables))+'\enskip = \enskip'+sp.latex(sp.nonlinsolve(ecuaciones,variables)))
            
            return
        
        
        input_matriz = st.radio('Cómo ingresar la matriz',
            ('Ingresar manualmente','Ingresar desde una lista')
        )


        if input_matriz == 'Ingresar manualmente':
            mtr_A = crear_matriz('A')
            mtr_B = crear_matriz('B')

            A = sp.Matrix(mtr_A)
            B = sp.Matrix(mtr_B)


    
        if input_matriz == 'Ingresar desde una lista':
        
            st.markdown(
                'Nota: El número de columnas de la matriz A debe ser igual al número de filas de la matriz B')

            st.markdown('Ingrese la matriz de la forma:')
            st.markdown('[[a11,a12,a13,...],[a21,a22,a23,...],[a31,a32,a33,...]]')

            st.markdown('Ejemplo: [[1,2,3],[4,5,6]]')
            st.info('Si A es de dimensiones n x m, B espera ser m x k.')

            A_col, B_col = st.columns(2)
            A_col.subheader('Matriz A')
            B_col.subheader('Matriz B')
            mtr_A = A_col.text_input(
                'Ingrese A:', '[[t1,t2,t3],[u,v,w],[x,y,z]]')
            mtr_B = B_col.text_input('Ingrese B:', '[[e,f,g],[h,r,p],[q,V,k]]')

            A = sp.Matrix(parse_expr(mtr_A))
            B = sp.Matrix(parse_expr(mtr_B))
            resolver_sistema(A,B)


        if opcion != 'Sistemas de Ecuaciones':

            if input_matriz == 'Ingresar manualmente':
                _,A_col, B_col,_ = st.columns([1,3,3,1])
                A_col.latex('A \;=\; '+sp.latex(A))
                B_col.latex('B \;=\; '+sp.latex(B))
              
            else:
                A_col.latex(tex_mtr(mtr_A))
                B_col.latex(tex_mtr(mtr_B, 'B'))
            
        
        opts = {
            'Sumar matrices':(0,'+'),
            'Restar matrices':(1,'-'), 
            'Multiplicar matrices':(2,"\cdot")}


        st.subheader(f'{opcion}')
        # refactoring made this more ilegible than it was before
        # but it's more efficient now, so it's worth it


        if opcion in opts.keys():
            dbool = bool(opcion == 'Multiplicar matrices')
            if (dbool and B.shape[1] != A.shape[0]):
                st.error('Las dimensiones de las matrices no coinciden')
                pass
            elif not dbool and B.shape[0] != A.shape[1]:
                st.error('Las dimensiones de las matrices no coinciden')
                pass
            else:
                st.latex(f'A {opts[opcion][1]} B \enskip = \enskip ' \
                +  sp.latex(operaciones[opts[opcion][0]](A, B))
                )

        if opcion == 'Determinante e Inversa':

            st.subheader('Inversa')
            try:
                st.latex('A^{-1} \enskip = \enskip '+sp.latex((A.inv()))) 
                st.latex('B^{-1} \enskip = \enskip '+sp.latex(B.inv()))
            except:
                st.error('Matriz no invertible')

            st.subheader('Determinante de una matriz')
            st.latex('det\,A \enskip = \enskip '+sp.latex(A.det()))
            st.latex('det\,B \enskip = \enskip '+sp.latex(B.det()))
            st.markdown('---')
            st.write('La determinante de un producto de matrices es igual a la multiplicación de sus determinantes')
            st.latex('det(AB) \enskip = \enskip '+sp.latex((A*B).det()))
            st.latex('det(A) \cdot det(B) \enskip = \enskip '+sp.latex(A.det()*B.det()))

        if opcion == 'Espacio Columna y Nulo':
            st.subheader('Espacio Nulo de una matriz')
            st.latex('N(A) \enskip = \enskip '+sp.latex(A.nullspace()))
            st.latex('C(A) \enskip = \enskip '+sp.latex(A.columnspace()))

            with st.expander('Más sobre espacio nulo:'):
                st.write('Definición')
                st.markdown(r'''Sea $T:V\rightarrow W $ una transformación lineal.''')
                st.markdown(r'''
                El núcleo, kernel o espanio nulo de $T$ es el conjunto de vectores en $V$ 
                que terminan en 0 de $W$ al pasar por $T$. Es decir,
                $$
                    N(T) = \{v \in V : T(v) = 0\}
                $$
                
                ''')


        if opcion == 'Producto Hadamard':
            st.subheader('Producto Hadamard')
            
            if A.shape != B.shape:
                st.error('Las dimensiones de las matrices no coinciden')
            else:
                try:
                    st.latex('A \circ B \enskip = \enskip ' +
                    sp.latex(sp.matrix_multiply_elementwise(A, B)))
                except:
                    st.info('Si una matriz contiene simbolos y la otra solo números,\
                        el producto falla (no debería)')
                    st.success('Solución: Agregar un simbolo cualquiera a la matriz \
                            que no contenga simbolos')
            
            with st.expander('Más sobre Hadamard'):
            
                st.write('A diferencia de la multiplicación de matrices, \
                    el producto Hadamard requiere que sean de las mismas dimensiones \
                    los objetos.')

                st.write('En los vectores se nota más su efecto, uno produce un escalar,\
                    hamadard otro vector del mismo tamaño, como multiplicar cada elemento por un escalar distinto.')
                st.latex(r'''
                c = \vec{x}^T \cdot \vec{y} =
            \begin{bmatrix}
                x_{1} \\
                x_{2} \\
                \vdots \\
                x_{n}
                \end{bmatrix}^T \cdot \begin{bmatrix}
                y_{1} \\
                y_{2} \\
                \vdots \\
                y_{n}
                \end{bmatrix} =
            x_1, x_2, \cdots, x_n] \cdot \begin{bmatrix}
                y_{1} \\
                y_{2} \\
                \vdots \\
                y_{n}
                \end{bmatrix} = \sum_{i=1}^n x_iy_i \in \mathbb{R}
                ''')
                st.latex(r'''\vec{z} = \vec{x} \odot \vec{y} =
                \begin{bmatrix}
                x_{1} \\
                x_{2} \\
                \vdots \\
                x_{n}
                \end{bmatrix} \odot \begin{bmatrix}
                y_{1} \\
                y_{2} \\
                \vdots \\
                y_{n}
                \end{bmatrix} =  \begin{bmatrix}
                x_1y_{1} \\
                x_2y_{2} \\
                \vdots \\
                x_ny_{n}
                \end{bmatrix} \in \mathbb{R}^n

                ''')

    def resolver_sistema(A,B):
        st.markdown('---')
        st.markdown('Sistema:')
        syms_dims = sp.symbols(
            ','.join([f'x_{i+1}' for i in range(A.shape[1])]))
        st.latex('Ax\;=\;b')
        st.latex(sp.latex(A*sp.Matrix(syms_dims)) +
                '\enskip = \enskip '+sp.latex(B))

        metodo = st.selectbox('Cómo quiere resolver el sistema',
                                ['Solución y ya', 'Gauss Jordan', 'Factorización LU'])

        if metodo == 'Solución y ya':
            st.latex('x \enskip = \enskip ' +
                        sp.latex(sp.linsolve((A, B), syms_dims)))

        if metodo == 'Gauss Jordan':
            st.latex('x\;=\;A^{-1}b')
            try:
                sols = A.gauss_jordan_solve(B)
                st.latex('x \; = \;' + sp.latex(A.inv()*B))
                if sols[1].shape[0] != 0:
                    st.latex('x \; = \;'+sp.latex(sols))
            except:
                st.error('Matriz no invertible')

        if metodo == 'Factorización LU':
            try:
                st.latex(f'x = {sp.latex(A.LUsolve(B))}')
            except:
                st.error('Matriz no invertible') 
   
    def_mtr()
    
if choice == "evaluador de matrices":   ###########################################3
     
    import streamlit as st
    import numpy as np
    import pandas as pd
    from numpy.linalg import LinAlgError
    import sympy as sp

    def det(matrix):
        return np.linalg.det(matrix)

    def inv(matrix):
        return np.linalg.inv(matrix)

    def rank(matrix):
        return np.linalg.matrix_rank(matrix)

    def transpose(matrix):
        return np.transpose(matrix)


    def mult(matrix1, matrix2):
        return np.dot(matrix1, matrix2)

    def gauss(matrix, b):
        return gauss_jordan(matrix, b)


    def gauss_jordan(matrix, b):
        n = len(matrix)
        augmented_matrix = np.hstack((matrix, b.reshape(-1, 1)))

        for i in range(n):
            pivot = augmented_matrix[i, i]
            if pivot == 0:
                for j in range(i + 1, n):
                    if augmented_matrix[j, i] != 0:
                        augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                        pivot = augmented_matrix[i, i]
                        break
            augmented_matrix[i] /= pivot
            for j in range(n):
                if j == i:
                    continue
                augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j, i]

        return augmented_matrix[:, -1]

    def calculate_result(matrix_a, scalar=None, matrix_b=None, operation="", b=None):
        if operation == "Suma":
            return matrix_a + matrix_b
        elif operation == "Resta":
            return matrix_a - matrix_b
        elif operation == "Multiplicación":
            return np.dot(matrix_a, matrix_b)
        elif operation == "División":
            return np.dot(matrix_a, np.linalg.inv(matrix_b))
        elif operation == "Determinante":
            return np.linalg.det(matrix_a)
        elif operation == "Inversa":
            return np.linalg.inv(matrix_a)
        elif operation == "Rango":
            return np.linalg.matrix_rank(matrix_a)
        elif operation == "Gauss-Jordan":
            return gauss_jordan(matrix_a, b)
        elif operation == "Transpuesta":
            return matrix_a.T
        elif operation == "Multiplicar por constante":
            return scalar * matrix_a
        else:
            raise ValueError("Operación no soportada")

    def display_equations_latex(matrix, b):
        rows, cols = matrix.shape
        equations = []
        symbols = [sp.Symbol(f"x_{j + 1}") for j in range(cols)]
        for i in range(rows):
            equation = 0
            for j in range(cols):
                equation += matrix[i, j] * symbols[j]
            eq = sp.Eq(equation, b[i])
            equations.append(sp.latex(eq))
        return equations

    def evaluate_polynomial(polynomial):
    # Crear un diccionario para almacenar las referencias a las matrices
        matrix_dict = {}

        for name, matrix in st.session_state.saved_matrices.items():
            if name in polynomial:
                if "det(" + name + ")" in polynomial:
                    matrix_dict[name] = np.linalg.det(matrix)
                    polynomial = polynomial.replace("det(" + name + ")", name)
                elif "transpose(" + name + ")" in polynomial:
                    matrix_dict[name] = matrix.T
                    polynomial = polynomial.replace("transpose(" + name + ")", name)
                else:
                    matrix_dict[name] = matrix

        # Agregar la función transpose() al espacio de nombres
        matrix_dict['transpose'] = np.transpose

        # Evaluar el polinomio usando el diccionario de matrices
        return eval(polynomial, matrix_dict)


    st.title("Evaluador de Matrices")

    operation = st.selectbox("Seleccione una operación", options=["Evaluador de Polinomios"])

    if operation == "Multiplicar por constante":
        scalar = st.number_input("Introduzca la constante", value=1.0)
    else:
        scalar = None

    st.write("## Matriz A")
    rows_a = st.number_input("Número de filas de la matriz A", min_value=1, value=2)
    cols_a = st.number_input("Número de columnas de la matriz A", min_value=1, value=2)
    matrix_a = np.zeros((rows_a, cols_a))

    b = np.zeros(rows_a)

    with st.container():
        input_columns_a_b = st.columns(cols_a + 1)
        for r in range(rows_a):
            for c in range(cols_a):
                matrix_a[r, c] = input_columns_a_b[c].number_input(f"A({r+1}, {c+1})", value=0.0, step=0.00001, format="%.5f", key=f"A_{r}_{c}_new")
        if operation == "Gauss-Jordan":
            for r in range(rows_a):
                b[r] = input_columns_a_b[cols_a].number_input(f"b({r+1})", value=0.0, step=0.00001, format="%.5f", key=f"b_{r}_new")

    if "saved_matrices" not in st.session_state:
        st.session_state.saved_matrices = {}

    st.header("Guardar Matrices")
    matrix_name = st.text_input("Nombre de la matriz (opcional)")
    save_matrix = st.selectbox("Seleccione una matriz para guardar", options=["Ninguna", "Matriz A"])

    if save_matrix == "Matriz A":
        if st.button("Guardar Matriz A"):
            st.session_state.saved_matrices[matrix_name] = matrix_a
            st.success(f"Matriz A guardada como '{matrix_name}'.")

    st.header("Cargar Matrices")
    load_matrix_name = st.selectbox("Seleccione una matriz guardada para cargar", options=["Ninguna"] + list(st.session_state.saved_matrices.keys()))
    if load_matrix_name != "Ninguna":
        load_matrix = st.selectbox("Seleccione la matriz donde cargar", options=["Ninguna", "Matriz A", "Matriz B"])
        if load_matrix != "Ninguna":
            load_button = st.button("Cargar Matriz")
            if load_button:
                if load_matrix == "Matriz A":
                    matrix_a = st.session_state.saved_matrices[load_matrix_name]
                elif load_matrix == "Matriz B" and operation in ["Suma", "Resta", "Multiplicación", "División"]:
                    matrix_b = st.session_state.saved_matrices[load_matrix_name]
                st.success(f"Matriz '{load_matrix_name}' cargada en {load_matrix}.")

    if operation == "Evaluador de Polinomios":
        st.subheader("Evaluador de Polinomios")
        
        # Previsualización de matrices cargadas
        st.subheader("Matrices guardadas")
        if st.session_state.saved_matrices:
            for name, matrix in st.session_state.saved_matrices.items():
                st.write(f"Matriz '{name}':")
                st.write(pd.DataFrame(matrix))
                
        else:
            st.write("No hay matrices guardadas.")
        
        st.write("""
        Por favor, introduzca el polinomio que desea evaluar. Recuerde usar los nombres de las matrices que ha guardado en este programa. 

        Para las operaciones, puede usar las siguientes funciones:

         - `det(matrix)`: Calcula el determinante de la matriz.
         - `inv(matrix)`: Calcula la matriz inversa.
         - `rank(matrix)`: Calcula el rango de la matriz.
         - `transpose(matrix)`: Transpone la matriz.
         - `mult(matrix1, matrix2)`: Multiplica dos matrices.
         - `gauss(matrix, b)`: Resuelve un sistema de ecuaciones lineales usando el método de Gauss-Jordan. Necesita una matriz y un vector.


        También puede usar los operadores básicos de Python:

        - `+` para la suma
        - `-` para la resta
        - `*` para la multiplicación
        - `/` para la división
        """)
        polynomial = st.text_input("Introduzca el polinomio a evaluar")

    try:
        if operation == "Evaluador de Polinomios":
            for matrix_name, matrix in st.session_state.saved_matrices.items():
                locals()[matrix_name] = matrix
            result = evaluate_polynomial(polynomial)
        elif operation in ["Suma", "Resta", "Multiplicación", "División"]:
            result = calculate_result(matrix_a, scalar, matrix_b, operation)
        elif operation == "Gauss-Jordan":
            result = calculate_result(matrix_a, scalar, operation=operation, b=b)
        else:
            result = calculate_result(matrix_a, scalar, operation=operation)

        st.subheader("Matriz A")
        st.write(pd.DataFrame(matrix_a))

        if operation == "Gauss-Jordan":
            st.subheader("Ecuaciones")
            equations = display_equations_latex(matrix_a, b)
            for eq in equations:
                st.latex(eq)

        if operation in ["Suma", "Resta", "Multiplicación", "División"]:
            st.subheader("Matriz B")
            st.write(pd.DataFrame(matrix_b))

        st.subheader("Resultado")
        if operation in ["Determinante", "Rango"]:
            st.write(result)
        elif operation == "Evaluador de Polinomios":
            if isinstance(result, np.ndarray):
                st.write(pd.DataFrame(result))
            else:
                st.write(result)
        else:
            st.write(pd.DataFrame(result))
    except ValueError as e:
        st.error(f"Error: {str(e)}")
    except LinAlgError as e:
        st.error(f"Error de álgebra lineal: {str(e)}")
    except NameError as e:
        st.error(f"Error de nombre: {str(e)}. Asegúrese de que las matrices en el polinomio existen y están guardadas.")
    except SyntaxError as e:
        st.error(f"Error de sintaxis: {str(e)}. Asegúrese de que el polinomio esté correctamente escrito.")

if choice == "minimos cuadrados":
    
    import streamlit as st
    import numpy as np
    from scipy.stats import pearsonr

    def generar_funcion_polinomial(coeficientes):
        grado = len(coeficientes) - 1
        variables = ['x^{}'.format(i) for i in range(grado, -1, -1)]
        coeficientes_str = ['{:.3f}'.format(coeficientes[i]) for i in range(grado, -1, -1)]

        funcion = ''
        for i in range(grado+1):
            funcion += coeficientes_str[i] + '*' + variables[i]
            if i != grado:
                funcion += ' + '
        
        return funcion

    def ajuste_polynomial(x, y):
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Las listas x e y no pueden estar vacías.")
        if len(x) != len(y):
            raise ValueError("Las listas x e y deben tener la misma longitud.")
        
        n = len(x)
        coeficientes_grados = []
        funciones_polinomiales = []
        coeficientes_correlacion = []

        for grado in range(1, n+1):
            matriz = np.vander(x, grado+1, increasing=True)
            coeficientes = np.linalg.lstsq(matriz, y, rcond=None)[0]
            coeficientes = np.round(coeficientes, 3)  # Redondear a 3 decimales
            funcion_polinomial = generar_funcion_polinomial(coeficientes)
            y_pred = np.dot(matriz, coeficientes)
            r, _ = pearsonr(y, y_pred)
            coeficientes_grados.append(coeficientes)
            funciones_polinomiales.append(funcion_polinomial)
            coeficientes_correlacion.append(r)

        return coeficientes_grados, funciones_polinomiales, coeficientes_correlacion

    st.title('Ajuste Polinomial y Coeficiente de Correlación')

    x_input = st.text_input("Ingrese los valores de x separados por espacios:")
    y_input = st.text_input("Ingrese los valores de y separados por espacios:")

    if x_input and y_input:
        try:
            x = np.array([float(valor) for valor in x_input.split()])
            y = np.array([float(valor) for valor in y_input.split()])
            coeficientes_grados, funciones_polinomiales, coeficientes_correlacion = ajuste_polynomial(x, y)

            for grado, (coeficientes, funcion_polinomial, r) in enumerate(zip(coeficientes_grados, funciones_polinomiales, coeficientes_correlacion), start=1):
                st.write(f"Grado {grado}:")
                st.write("Coeficientes:", coeficientes)
                st.latex("Función polinomial: " + funcion_polinomial)
                st.write("Coeficiente de correlación:", r)
        except ValueError as e:
            st.error(str(e))

