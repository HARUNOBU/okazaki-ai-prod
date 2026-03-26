配置先:
- config.py                  -> /opt/okazaki-ai-portal/current/src/config.py
- それ以外の *.py           -> /opt/okazaki-ai-portal/current/src/tools/

置換後の確認:
python -m py_compile /opt/okazaki-ai-portal/current/src/config.py /opt/okazaki-ai-portal/current/src/tools/*.py
