"""Chat format utils."""
from typing import Optional

from megatron.core import mpu

def make_context(
        message: Optional[str], history: list, tokenizer,
        max_window_size: int,
        system: str = '',
        response_prefix: str = '',
        chat_format: str = 'human-assistant',
        verbose: bool = True,
):
    if mpu.get_tensor_model_parallel_rank() != 0:
        raw_text = ''
        context_tokens = tokenizer.tokenize('EMPTY TEXT')
        raw_text_len = 0
        context_length = 0
        return raw_text, context_tokens, raw_text_len, context_length

    if chat_format in ['human-assistant', 'human-assistant-llama']:
        raw_text = ''
        for item in history[::-1]:
            query, response = item['user'], item['bot']
            prev_chat = f'\n\nHuman: {query}\n\nAssistant: {response}'
            current_context_size = len(tokenizer.tokenize(raw_text)) + \
                                   len(tokenizer.tokenize(prev_chat))
            if current_context_size < max_window_size:
                raw_text = prev_chat + raw_text
            else:
                break
        if message is not None:
            raw_text += f'\n\nHuman: {message}\n\nAssistant:'
        context_tokens = tokenizer.tokenize(raw_text)

        # LLaMA: add BOS.
        if chat_format == 'human-assistant-llama':
            bos = tokenizer.detokenize([tokenizer.bos_id])
            raw_text = bos + raw_text
            context_tokens = [tokenizer.bos_id] + context_tokens
    elif chat_format == "raw":
        raw_text = ''
        for item in history[::-1]:
            query, response = item['user'], item['bot']
            prev_chat = f'\n\n{query}\n\n{response}'
            current_context_size = len(tokenizer.tokenize(raw_text)) + \
                                   len(tokenizer.tokenize(prev_chat))
            if current_context_size < max_window_size:
                raw_text = prev_chat + raw_text
            else:
                break
        if message is not None:
            raw_text += f'{message}'
        context_tokens = tokenizer.tokenize(raw_text)

    elif chat_format in ['chatml', 'chatml-llama', 'chatml-nosys']:
        im_start, im_end = '<|im_start|>', '<|im_end|>'
        im_start_tokens = tokenizer.tokenize(im_start)
        im_end_tokens = tokenizer.tokenize(im_end)
        nl_tokens = tokenizer.tokenize('\n')

        def _tokenize_str(role, content):
            return f'{role}\n{content}', tokenizer.tokenize(role) + nl_tokens + tokenizer.tokenize(content)

        if system != '':
            system_text, system_tokens_part = _tokenize_str('system', system)
            system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
        else:
            system_text = ""
            system_tokens = []

        raw_text = ''
        context_tokens = []

        for item_idx, item in enumerate(history[::-1]):
            query, response = item['user'], item['bot']
            query_text, query_tokens_part = _tokenize_str('user', query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str('assistant', response)
            response_tokens = im_start_tokens + response_tokens_part

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = f'\n{im_start}{query_text}{im_end}\n{im_start}{response_text}'
            if (message is not None) or (item_idx > 0):
                next_context_tokens += im_end_tokens
                prev_chat += f'{im_end}'

            # chatml-nosys: remove the first '\n' of next_context_tokens, so size == -1.
            _system_tokens_size = -1 if chat_format == 'chatml-nosys' else len(system_tokens)
            current_context_size = _system_tokens_size + len(next_context_tokens) + len(context_tokens)
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        if chat_format == 'chatml-nosys':
            # chatml-nosys: remove the first '\n'.
            if context_tokens:
                assert context_tokens[:len(nl_tokens)] == nl_tokens
                assert raw_text.startswith('\n')
                context_tokens = context_tokens[len(nl_tokens):]
                raw_text = raw_text[1:]
        else:
            context_tokens = system_tokens + context_tokens
            raw_text = f'{im_start}{system_text}{im_end}' + raw_text

        if message is not None:
            context_tokens += (
                    nl_tokens + im_start_tokens + _tokenize_str('user', message)[1] + im_end_tokens +
                    nl_tokens + im_start_tokens + tokenizer.tokenize('assistant') + nl_tokens
            )
            raw_text += f'\n{im_start}user\n{message}{im_end}\n{im_start}assistant\n'

        if response_prefix != "":
            context_tokens += tokenizer.tokenize(response_prefix)
            raw_text += response_prefix

        # LLaMA: add BOS.
        if chat_format == 'chatml-llama':
            bos = tokenizer.detokenize([tokenizer.bos_id])
            raw_text = bos + raw_text
            context_tokens = [tokenizer.bos_id] + context_tokens
    else:
        raise NotImplementedError(f'Unknown chat format {chat_format!r}')
    if verbose:
        print('\nSystem:', system)
        print('\nContext:', raw_text)
    raw_text_len = len(raw_text)
    context_length = len(context_tokens)
    return raw_text, context_tokens, raw_text_len, context_length

