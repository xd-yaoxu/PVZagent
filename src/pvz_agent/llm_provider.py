"""Minimal user-configured compatible API client. No game input or credential logs."""
import json
import os
import urllib.error
import urllib.parse
import urllib.request


class ProviderError(Exception):
    def __init__(self,code,http_status=None):
        self.code=code;self.http_status=http_status
        super().__init__(code+(f' (HTTP {http_status})' if http_status else ''))


def load_local_api_key(name='PVZ_LLM_API_KEY'):
    value=os.environ.get(name,'').strip()
    if not value and os.name=='nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as key:value=str(winreg.QueryValueEx(key,name)[0]).strip()
        except FileNotFoundError:pass
    if not value:raise ProviderError('missing_local_api_key')
    return value


def http_transport(url,body,key,timeout):
    request=urllib.request.Request(url,data=json.dumps(body,ensure_ascii=False,allow_nan=False).encode('utf-8'),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(request,timeout=timeout) as response:
            raw=response.read(2*1024*1024+1)
    except urllib.error.HTTPError as exc:raise ProviderError('provider_http_error',exc.code) from None
    except Exception as exc:
        timed_out=isinstance(exc,TimeoutError) or isinstance(getattr(exc,'reason',None),TimeoutError)
        raise ProviderError('provider_timeout' if timed_out else 'provider_transport_error') from None
    if len(raw)>2*1024*1024:raise ProviderError('provider_response_too_large')
    try:return json.loads(raw)
    except (ValueError,UnicodeError):raise ProviderError('provider_invalid_json') from None


def extract_text(response,protocol):
    if protocol=='responses':
        if response.get('status') not in (None,'completed'):raise ProviderError('response_not_completed')
        chunks=[]
        for item in response.get('output',[]):
            if item.get('type')!='message':continue
            for content in item.get('content',[]):
                if content.get('type')=='refusal':raise ProviderError('model_refusal')
                if content.get('type')=='output_text':chunks.append(content.get('text',''))
        text=''.join(chunks)
    else:
        try:
            choice=response['choices'][0];message=choice['message']
            if choice.get('finish_reason') not in (None,'stop'):raise ProviderError('response_not_completed')
            if message.get('refusal'):raise ProviderError('model_refusal')
            text=message.get('content','')
        except (KeyError,IndexError,TypeError):raise ProviderError('provider_response_shape_invalid') from None
    if not isinstance(text,str) or not text.strip():raise ProviderError('provider_empty_text')
    return text


class CompatibleProvider:
    def __init__(self,config,*,transport=http_transport,key_reader=load_local_api_key):
        self.config=dict(config);self.transport=transport;self.key_reader=key_reader
        parsed=urllib.parse.urlsplit(self.config['base_url'])
        if parsed.scheme!='https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('HTTPS base URL without embedded credentials/query required')
        if not isinstance(config.get('model'),str) or not config['model'].strip():raise ValueError('Exact model ID required')
        # Historical Sol configurations are preserved for provenance, not live calls.
        if config['model'].strip().lower().replace('-','').replace('_','') == 'gpt6.1sol':
            raise ValueError('Historical Sol model disabled; use scripts/run_product_agent.py for the current product model')
        if config.get('protocol') not in ['auto','responses','chat_completions']:raise ValueError('Unsupported protocol')
        self.protocol=None if config['protocol']=='auto' else config['protocol']

    def _call(self,protocol,prompt,*,image_data_url=None,json_schema=None,max_tokens=None):
        if image_data_url and not self.config.get('image_transfer_authorized',False):
            raise ProviderError('image_transfer_not_authorized')
        tokens=max_tokens or self.config['max_output_tokens'];model=self.config['model']
        if protocol=='responses':
            content=[dict(type='input_text',text=prompt)]
            if image_data_url:content.append(dict(type='input_image',image_url=image_data_url))
            body=dict(model=model,input=[dict(role='user',content=content)],max_output_tokens=tokens,store=False)
            if json_schema:body['text']=dict(format=dict(type='json_schema',name='pvz_readonly_advice',strict=True,schema=json_schema))
            suffix='/responses'
        else:
            content=[dict(type='text',text=prompt)]
            if image_data_url:content.append(dict(type='image_url',image_url=dict(url=image_data_url)))
            body=dict(model=model,messages=[dict(role='user',content=content)],max_completion_tokens=tokens)
            if json_schema:body['response_format']=dict(type='json_schema',json_schema=dict(name='pvz_readonly_advice',strict=True,schema=json_schema))
            suffix='/chat/completions'
        key=self.key_reader(self.config['api_key_environment_variable'])
        response=self.transport(self.config['base_url'].rstrip('/')+suffix,body,key,self.config['timeout_seconds'])
        text=extract_text(response,protocol)
        usage=response.get('usage') or {}
        return dict(text=text,response_id=response.get('id'),protocol=protocol,model_returned=response.get('model'),
            usage={k:v for k,v in usage.items() if k in ['input_tokens','output_tokens','total_tokens','prompt_tokens','completion_tokens'] and type(v) is int})

    def probe(self):
        protocols=[self.protocol] if self.protocol else ['responses','chat_completions']
        attempts=[]
        for protocol in protocols:
            try:
                result=self._call(protocol,'Return exactly OK.',max_tokens=64)
                self.protocol=protocol;attempts.append(dict(protocol=protocol,status='completed'))
                return dict(protocol=protocol,attempts=attempts,result=result)
            except ProviderError as exc:
                attempts.append(dict(protocol=protocol,error=exc.code,http_status=exc.http_status))
                # Do not retry authentication, quota, timeouts, or other server errors.
                if protocol!=protocols[-1] and exc.http_status in [404,405,501]:continue
                exc.attempts=list(attempts)
                raise
        raise ProviderError('no_compatible_protocol')

    def complete(self,prompt,**kwargs):
        if self.protocol is None:raise ProviderError('protocol_probe_required')
        return self._call(self.protocol,prompt,**kwargs)
