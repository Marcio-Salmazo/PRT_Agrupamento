----------------------------------------------------------------------------------------------------------

Implementação do algorítmo k-means - Prática 1

Grupo:
Andrey dos Reis Cadima Dias - Matrícula: 1241XESP054
Marcio Salmazo Ramos - Matrícula: 12412CCP021
Pedro Victor Guerra de Figueiredo - Matrícula: 1232XESP019

21/05/2024

----------------------------------------------------------------------------------------------------------

Instruções de uso:

-> O arquivo contendo a implementação do algorítmo é denominado 'k-means Scratch NP.ipynb', 
-> É necessário utilizar o jupyter notebook para abrir o programa e executá-lo

	* Importante salientar que o Jupyter notebook utilizado por meio da IDE VSCODE, 
	  o procedimento para instalar e utilizar o Jupyter notebook pode ser encontrado 
	  no seguinte link: https://www.alphr.com/vs-code-open-jupyter-notebook/

-> Antes de iniciar a execução do algorítmo, é necessário realizar a instalação de algumas
   bibliotecas. Segue a baixo o comando para instalar tais pacotes dentre do Jupyter Notebook:

	* pip install tk
        * pip install scikit-learn
        * pip install pandas
        
	> OBS: As demais bibliotecas utilizadas (Math e Numpy) já são comumente instaladas
	  junto com o Python, tornando a instalação desnecessário

-> Após a instalação das bibliotecas, cada célula de código do algorítmo deve ser executada
   EM ORDEM, também há a opção no VSCODE para executar todas as celulas em sequência.

-> A terceira célula do notebook (responsável pela importação do dataset) EXIGE duas entradas
   do usuário:

	* Primeiramente será necessário selecionar o arquivo (em formato .CSV) contendo o dataset
	  Uma nova janela será aberta solicitando a seleção do arquivo
	* Após a seleção do arquivo será solicitado ao usuário que insira o nome da coluna de classe,
	  a fim de que ela seja excluída do processo (É importante salientar que a entrada é CASE-SENSITIVA) 
		
		Obs: Caso o arquivo não contenha um atributo classe, basta clicar 'ESC' ao surgir a solicitação
		     ou é possível inserir um nome aleátorio (que não exista no dataset)

OBS: O processo de normalização dos dados apenas vai executar se, e somente se TODOS os valores
     atributos do dataset forem NUMÉRICOS (int64 ou float64)

-> A execução das demais células do algorítmo vão apresentar os resultados dentro da própria IDE,
   com isso, é possível ir acompanhando o pré-processamento dos dados antes da sua aplicação no
   k-means (caso necessário).

-> Ao final da execução, o arquivo 'Resultado.csv' será criado (no mesmo ambiente que contém o aquivo .ipynb)
   Tal arquivo contem os dados originais do dataset inserido, acrescido do atributo 'Grupo' o qual descreve os
   grupos associados à cada objeto/registro  
	
----------------------------------------------------------------------------------------------------------




