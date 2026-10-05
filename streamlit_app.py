"""Run: streamlit run streamlit_app.py"""
import string
import re
from pathlib import Path
import streamlit as st
from config import PROJECT_ROOT, MANIFEST_PATH
from translation import TextPreprocessor, SignVocabulary, plan_coverage
from playback import export_sentence, cache_key


def main():
    st.set_page_config(page_title='NSL Text-to-Sign', page_icon='🤟', layout='wide')
    st.title('Nigerian Sign Language · Text to Sign')
    try:
        vocab = SignVocabulary()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return
    reference_letters = {p.stem: p for p in (PROJECT_ROOT / 'assets/asl_demo').glob('*.png')
                         if len(p.stem) == 1 and p.stem in string.ascii_uppercase}
    # Registered NSL recordings take priority over the reference illustrations.
    vocab.alphabet = {**reference_letters, **vocab.alphabet}
    with st.sidebar:
        st.header('Available recordings')
        st.metric('Vocabulary labels', len(vocab.index))
        st.metric('Fingerspelling letters', f'{len(vocab.alphabet)}/26')
        skeleton_letters = sorted(c for c, p in vocab.alphabet.items() if p.with_suffix('.pkl').is_file())
        st.caption(f'Recorded finger skeletons: {len(skeleton_letters)}/26')
        unavailable_skeletons = sorted(set(string.ascii_uppercase) - set(skeleton_letters))
        if unavailable_skeletons:
            st.caption('Stickman letters awaiting recordings: ' + ', '.join(unavailable_skeletons))
        if not vocab.curated:
            st.warning('These recordings have not been screened for technical quality yet.')
        with st.expander('Browse vocabulary'):
            search = st.text_input('Find a sign').upper().replace(' ', '_')
            st.write(', '.join(k.replace('_', ' ') for k in sorted(vocab.index) if search in k))
        missing_letters = sorted(set(string.ascii_uppercase) - set(vocab.alphabet))
        if missing_letters:
            st.caption('Letters awaiting recordings: ' + ', '.join(missing_letters))
    with st.form('translate'):
        text = st.text_area('English text', placeholder='Hello', max_chars=300)
        spell_only = st.checkbox('Fingerspell every word', value=False)
        mode_label = st.radio('Playback', ['Original recording', 'Stickman animation'], horizontal=True)
        speed = st.select_slider('Playback speed', options=[0.25, 0.5, 0.75, 1.0], value=1.0, format_func=lambda value: f'{value:g}×')
        close_up = st.checkbox('Enlarge signer in stickman animation', value=True)
        submitted = st.form_submit_button('Translate', type='primary')
    if submitted:
        st.session_state.pop('translation', None)
        try:
            tokens = re.findall(r'[A-Z0-9]+', text.upper()) if spell_only else TextPreprocessor().preprocess(text)
        except (RuntimeError, ImportError) as error:
            st.error(f'English preprocessing unavailable: {error}')
            return
        if not tokens:
            st.warning('Enter some text to translate.')
            return
        if spell_only:
            illustrated = SignVocabulary(alphabet_dir=PROJECT_ROOT / 'assets/asl_demo')
            plan = illustrated.fingerspell(tokens)
        else:
            plan = vocab.resolve_tokens(tokens)
        fuzzy_matches = [e for e in plan if e.get('match_type') == 'fuzzy']
        if fuzzy_matches:
            st.caption('Similar-spelling matches: ' + '; '.join(
                f"{e['requested_gloss']} → {e['matched_gloss']} ({e['confidence']:.1f}/100 string similarity)"
                for e in fuzzy_matches))
        coverage = plan_coverage(plan)
        st.write('Playback order: ' + ' → '.join(e['gloss'].replace('_', ' ') for e in plan))
        st.caption(f"Direct sign coverage: {coverage['direct_coverage']:.0%} · Playable coverage: {coverage['playable_coverage']:.0%}")
        unavailable = [e for e in plan if e['type'] == 'missing']
        if unavailable:
            st.error('A complete translation is unavailable for: ' + ', '.join(e['gloss'] for e in unavailable))
            st.write('Missing fingerspelling characters: ' + ', '.join(sorted({c for e in unavailable for c in e['missing']})))
            return
        mode = 'source' if spell_only or mode_label == 'Original recording' else 'stickman'
        if spell_only:
            st.caption('Fingerspelling displays illustrated letter cards. Playback style applies to recorded signs.')
        try:
            with st.spinner('Preparing ordered playback…'):
                output = PROJECT_ROOT / 'results/video_cache' / (cache_key(plan, mode, speed, close_up) + '.mp4')
                if not output.exists():
                    export_sentence(plan, output, mode, speed, close_up)
            st.session_state['translation'] = {'path': str(output), 'text': text, 'reference_letters': any(Path(path).parent == PROJECT_ROOT / 'assets/asl_demo' for entry in plan if entry['type'] == 'fingerspell' for _, path in entry['letters'])}
        except (ValueError, OSError, RuntimeError) as error:
            st.error(f'Could not render this sentence: {error}')
    if 'translation' in st.session_state:
        result = st.session_state['translation']
        st.write(result['text'])
        st.video(result['path'])
        with open(result['path'], 'rb') as handle:
            st.download_button('Download video', handle.read(), 'nsl-translation.mp4', 'video/mp4')


if __name__ == '__main__':
    main()
