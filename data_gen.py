import pathlib
from urllib.parse import unquote

import pandas as pd

base_dir = pathlib.Path(r"C:\Users\yfq61\Documents\project\log_data2")
csv_list = list(base_dir.glob("*.csv"))
test_file = csv_list[0]
one_file_limit = 1000000
columns = ['用户ip', '时间', '响应状态码', '包体长度', '访问的方法', '访问的路径', '访问的参数', '跳转地址', '用户请求头', '攻击标签']
attack_type_list = ['SQL注入', '远程代码执行', 'XSS', 'webshell', '漏洞攻击']
attack_map = {'SQL注入攻击': 'SQL注入', 'SQL盲注攻击探测': 'SQL注入', 'mssql注入攻击': 'SQL注入', 'sqlmap黑客工具': 'SQL注入', 'sql注入攻击': 'SQL注入',
              'Struts2远程代码执行攻击': '远程代码执行', 'linux 敏感命令执行': '远程代码执行', 'windows 敏感命令执行': '远程代码执行',
              'xss攻击': 'XSS', '跨站脚本攻击(XSS)': 'XSS',
              '蚁剑webshell管理工具': 'webshell',
              'Jndi注入': '漏洞攻击', 'LDAP漏洞攻击': '漏洞攻击', 'swfupload跨站': '漏洞攻击', 'web协议攻击': '漏洞攻击', '文件包含漏洞': '漏洞攻击',
              '漏洞特殊字符fuzzy': '漏洞攻击', '疑似远程文件加载': '漏洞攻击', '超长字符串': '漏洞攻击'}


def map_attack_label(x):
    if x in attack_map:
        return attack_type_list.index(attack_map[x]) + 1
    else:
        return len(attack_type_list) + 1


def split_to_black_and_white(in_dir, out_dir):
    white_i = 0
    black_i = 0
    df_white = pd.DataFrame()
    df_black = pd.DataFrame()
    for csv_file in in_dir.glob("*.csv"):
        print(csv_file)
        df_tmp = pd.read_csv(csv_file, names=columns)
        df_tmp['访问的方法'] = df_tmp['访问的方法'].fillna("-")
        df_tmp['访问的参数'] = df_tmp['访问的参数'].fillna("-")
        df_tmp['跳转地址'] = df_tmp['跳转地址'].fillna("-")
        df_tmp['用户请求头'] = df_tmp['用户请求头'].fillna("-")
        df_tmp['全文本'] = df_tmp['用户ip'].map(str) + ',' + df_tmp['时间'].map(str) + ',' + df_tmp['响应状态码'].map(str) + ',' + \
                        df_tmp[
                            '包体长度'].map(str) + ',' + df_tmp['访问的方法'].map(str) + ',' + df_tmp['访问的路径'].map(str) + ',' + \
                        df_tmp['访问的参数'].map(str) + ',' + \
                        df_tmp['跳转地址'].map(str) + ',' + df_tmp['用户请求头'].map(str)
        # df_tmp['全文本'] = df_tmp['全文本'].apply(unquote)
        df_tmp['攻击标签'] = df_tmp['攻击标签'].fillna("[]")
        df_tmp = df_tmp[['攻击标签', '全文本']]
        df_tmp_white = df_tmp[df_tmp['攻击标签'] == "[]"].copy()
        df_tmp_black = pd.DataFrame(
            [[a, b] for A, b in
             df_tmp[df_tmp['攻击标签'] != "[]"].itertuples(index=False) for a in
             eval(A)], columns=df_tmp.columns)

        df_tmp_white['攻击标签'] = 0
        df_tmp_black = df_tmp_black[(df_tmp_black['攻击标签'] != '敏感目录') & (df_tmp_black['攻击标签'] != '敏感接口api')]
        df_tmp_black['攻击标签'] = df_tmp_black['攻击标签'].map(map_attack_label)
        df_tmp_black = df_tmp_black[(df_tmp_black['攻击标签'] != len(attack_type_list) + 1)]

        df_white = pd.concat([df_white, df_tmp_white])
        df_black = pd.concat([df_black, df_tmp_black])
        while len(df_white) > one_file_limit:
            df_white[:one_file_limit].to_csv(out_dir.joinpath(f'white-{white_i}.csv'), index=False, escapechar="\\")
            white_i += 1
            df_white = df_white[one_file_limit:]
        while len(df_black) > one_file_limit:
            df_black[:one_file_limit].to_csv(out_dir.joinpath(f'black-{black_i}.csv'), index=False, escapechar="\\")
            black_i += 1
            df_black = df_black[one_file_limit:]
    df_black.to_csv(out_dir.joinpath(f'black-{black_i}.csv'), index=False, escapechar="\\")
    df_white.to_csv(out_dir.joinpath(f'white-{white_i}.csv'), index=False, escapechar="\\")

def conncat_white_to_one_file(root_path):
    with open(root_path.joinpath("white.csv"), "w", encoding="utf-8") as f:
        for files in root_path.glob("white-*.csv"):
            with open(files, "r", encoding="utf-8") as f2:
                f.write(f2.read())

def del_colunms_in_out_files(file_path):
    df = pd.read_csv(file_path)
    df = df[df['攻击标签'] != '攻击标签']
    df.to_csv(file_path)

if __name__ == "__main__":
    out_dir = pathlib.Path(r"bigdata")
    # if not out_dir.exists():
        # out_dir.mkdir()
    # split_to_black_and_white(base_dir, out_dir)

    # conncat_white_to_one_file(out_dir)
    del_colunms_in_out_files(out_dir.joinpath("white.csv"))
