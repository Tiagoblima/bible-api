# bible-api
The project aligns verses of the bible from diferente version according to the verses and also drops the duplicate verses. 

       parser.add_argument('--input_dir', type=str,
                        help='O ano de backup', required=True)
       parser.add_argument('--output_dir', type=str, default=None,
                        help='O ano de backup', required=False)
       parser.add_argument('--from_sqlite', action="store_true",
                        help='Não reseta o banco antes da modificação')
                        
                        
--input_dir diretório com os dados originais em csv

--output_dir diretório de saída dos arquivos processados

--from_sqlite indica se está carregando arquivos sqlite
