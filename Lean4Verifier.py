
  
import subprocess
import tempfile
import re

SORRY_ERROR = "/tmp/tmpnone_dz:{line_num}:0: error: incomplete prove with 'sorry' placeholder"


class Lean4Verifier:
    def __init__(self, code: str, position: str='after', code_type = 'proof'): #question or proof
        self._code = code

        self._cleaned_code = None
        self._error_annotated_code = None
        self._code_type = code_type
        self._position = position
        
        self._error_msg = None
        self._raw_error_msg = None
        
        if self._code == None:
            self._if_correct = False
            self._verified = True
        else:
            self._if_correct = None
            self._verified = False
    
    def _process_error_messages(self, message: list) -> None:
        if 'error' in message:
            
            self._if_correct = False
            
            comments_to_add = []
            
            # errors = message.split('\n')
            
            errors = re.findall(r'/tmp[^:\n]+:[\d:]+:.*?(?=(?:/tmp|\Z))', message, re.DOTALL)
            
            errors = [e for e in errors if 'error' in e]
            
            self._error_msg = '\n'.join(errors)
            
            
            
            # print(errors)
            for error in errors:
                # print("######")
                # print(error)
                line_num = int(error.split(' ')[0].strip().split(':')[-3]) - 1
                # print("######")
                
                
                if self._position == 'before':
                    comments_to_add.append((line_num, f"/- Error in the following line: {error} -/"))
                    
                elif self._position == 'after':
                    comments_to_add.append((line_num + 1, f"/- {error} -/"))

                else:
                    print("please set position to 'before' or 'after'")

            self._error_annotated_code = self._code.splitlines()

            for line_num, comment in sorted(comments_to_add, key=lambda x: x[0], reverse=True):
                self._error_annotated_code.insert(line_num, comment)  

            self._error_annotated_code = '\n'.join(self._error_annotated_code)
                
            
        else:
            self._if_correct = True
    
    def _remove_lean4_comments(self) -> None:
        """Remove both block comments (/- ... -/) and line comments (--) from Lean4 code."""
        # Remove block comments
        cleaned_code = self._code
        while (start := cleaned_code.find('/-')) != -1:
            if (end := cleaned_code.find('-/', start)) == -1:
                break
            cleaned_code = f"{cleaned_code[:start]}{cleaned_code[end + 2:]}"
        
        # Remove line comments and empty lines
        self._cleaned_code = '\n'.join(
            line for line in cleaned_code.splitlines()
            if not line.strip().startswith('--')
        )

    def _verify(self) -> None:
     
        with tempfile.NamedTemporaryFile(mode='w+', encoding='utf-8') as temp_file:
            temp_file.write(self._code)
            temp_file.seek(0)
            process = subprocess.Popen(["lake", "env", "lean",temp_file.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()    
            stdout_str = stdout.decode('utf-8').strip()

            self._raw_error_msg = stdout
            
            # print(stdout)
            # print(stderr)
            if self._code_type == 'proof':
                # remove comments and detect sorry
                self._remove_lean4_comments()
                if 'sorry' in self._cleaned_code:

                    sorry_line_num = []
                    
                    # detectline
                    line_list = self._code.split('\n')
              
                    for i in range(len(line_list)):
                        if 'sorry' in line_list[i].strip():
                            sorry_line_num.append(i+1)
                
                    # define error messaage
                    for n in sorry_line_num:
                        # print('i')
                        stdout_str += f'\n{SORRY_ERROR.format(line_num=n)}'
                
            # print(stdout_str)
            if stdout_str =='':
                self._if_correct = True
            else:
                self._process_error_messages(stdout_str)
            
        self._verified = True
        
    @property
    def if_correct(self) -> bool:
        # print('dd')
        if not self._verified:
            self._verify()
        return self._if_correct
    
    @property
    def error_message(self) -> list:
        if not self._verified:
            self._verify()
        return self._error_msg
    
    @property
    def error_annotated_code(self) -> list:
        if not self._verified:
            self._verify()
        return self._error_annotated_code
    
    @property
    def raw_error_msg(self):
        if not self._verified:
            self._verify()
        return self._raw_error_msg




