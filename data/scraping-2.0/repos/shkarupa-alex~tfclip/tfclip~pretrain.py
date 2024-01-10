from tfclip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, INCEPTION_MEAN, INCEPTION_STD, \
    IMAGENET_MEAN, IMAGENET_STD


def _pcfg(version='1.0.0', sha256=None, **kwargs):
    # OpenAI / OpenCLIP defaults
    return {
        'mean': OPENAI_DATASET_MEAN,
        'std': OPENAI_DATASET_STD,
        'interpolation': 'bicubic',
        'resize_mode': 'shortest',
        'version': version,
        'sha256': sha256,
        **kwargs,
    }


def _slpcfg(version='1.0.0', sha256=None, **kwargs):
    # SiGLIP defaults
    return {
        'mean': INCEPTION_MEAN,
        'std': INCEPTION_STD,
        'interpolation': 'bicubic',
        'resize_mode': 'squash',
        'version': version,
        'sha256': sha256,
        **kwargs,
    }


def _apcfg(version='1.0.0', sha256=None, **kwargs):
    # CLIPA defaults
    return {
        'mean': IMAGENET_MEAN,
        'std': IMAGENET_STD,
        'interpolation': 'bilinear',
        'resize_mode': 'squash',
        'version': version,
        'sha256': sha256,
        **kwargs,
    }


_VITB32 = dict(
    # openai=_pcfg(),  # OpenAI models were trained with QuickGELU
    laion400m_e31=_pcfg(sha256='320662fda288042c60a3b6175fb4c31f7b71803d9ce6d6ceb1fd69f5c9340b76'),
    laion400m_e32=_pcfg(sha256='868b1597cc633a9ed2a0aecb07fc77b1a939b6ef11a719bfdacbab19f524fe12 '),
    laion2b_e16=_pcfg(sha256='649b1d8ed57828bebd1256b7bd81e51956cb2a9fa6f882fa20abc216811b8917'),
    laion2b_s34b_b79k=_pcfg(sha256='c9b3aa9965d2dc9d34fe9751ae3b8b00b56cf30bcf260ffab2cf9c7cb6ecc38a'),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(sha256='d550a54e17fc9a9af9bbe620ce48ae7c20f8d38761f6e5467a2be1b7688fe60f'),
    # DataComp-M models
    datacomp_m_s128m_b4k=_pcfg(sha256='b081e03fa06c4dec5ae383887b2072ad22197ce47a74b5d668a136c313aa91d1'),
    commonpool_m_clip_s128m_b4k=_pcfg(sha256='807883587d8a54e5aad62268f9736e288ed8dabb9fdaeb1562382d9623c3fea7'),
    commonpool_m_laion_s128m_b4k=_pcfg(sha256='7062564cb76671d629018f203114664a34039d2153cfc9dcf1150023b2acd6b5'),
    commonpool_m_image_s128m_b4k=_pcfg(sha256='0b2baccc96cd6717bf945bd6004bc552cee528461e861e8a53b9f1d3ad40f99a'),
    commonpool_m_text_s128m_b4k=_pcfg(sha256='ae9d94b8ff4083be9798840d474478fb2102f2f93e6125710684e46543261057'),
    commonpool_m_basic_s128m_b4k=_pcfg(sha256='b105cef0763b8aea4fd73b6526f2392180a41c485646f0704fd3386092905e75'),
    commonpool_m_s128m_b4k=_pcfg(sha256='c6cd5fcac6664ea520baa38337f769ba3f8516cd551f0b90511cc83be63415e2'),
    # DataComp-S models
    datacomp_s_s13m_b4k=_pcfg(sha256='cdc5303490ed6df01475ac0ccedb3435ae9ead16d22edcc25c65492e573d5a98'),
    commonpool_s_clip_s13m_b4k=_pcfg(sha256='35e3073a991db2cf92129e71da6aae98da560f268d867199d5ed7cc07abeac53'),
    commonpool_s_laion_s13m_b4k=_pcfg(sha256='aca756a042efffdd9f2f1ae64cfb51dcb8cff1ccec15bae1bbcb2db735e85bf5'),
    commonpool_s_image_s13m_b4k=_pcfg(sha256='cdc5303490ed6df01475ac0ccedb3435ae9ead16d22edcc25c65492e573d5a98'),
    commonpool_s_text_s13m_b4k=_pcfg(sha256='ecb478df74139d50fea4602dc7f2852b6f0722d83583ff22ab537a3e80939a79'),
    commonpool_s_basic_s13m_b4k=_pcfg(sha256='d74f3485d2a59b32b8a8bc44524d62ea0e01596cdd8d2240f01934072cc983de'),
    commonpool_s_s13m_b4k=_pcfg(sha256='d4333ca7b35e67444f18c57e35f5c2de8e17ee76f859fc1efd5bfa13fa0ad776'),
)

_VITB32_quickgelu = dict(
    openai=_pcfg(sha256='2e0f33c7e468ddaf7e5f615ad557faaaca16ba1f1cf49558953f8bda6cd16006'),
    laion400m_e31=_pcfg(sha256='320662fda288042c60a3b6175fb4c31f7b71803d9ce6d6ceb1fd69f5c9340b76'),
    laion400m_e32=_pcfg(sha256='868b1597cc633a9ed2a0aecb07fc77b1a939b6ef11a719bfdacbab19f524fe12'),
    metaclip_400m=_pcfg(sha256='57a88f153fbae708ef901ecf2ad8288dad702f5693793515f0371fdcdf480e50'),
    metaclip_fullcc=_pcfg(sha256='cce1320dacdf0b44d2c768e62518b7e37281ff264cfea64f986a6eeb586fe955'),
)

_VITB32_256 = dict(
    datacomp_s34b_b86k=_pcfg(sha256='24aba612bd1d6b50908be3f607c685a1b6a2b962f2d38dfec5dec69fd5cd1d83'),
)

_VITB16 = dict(
    # openai=_pcfg(),  # OpenAI models were trained with QuickGELU
    laion400m_e31=_pcfg(sha256='14796113406aaf645350d0258b937a2e5729d5c851b6b9bed4c5d4b8aefe124f'),
    laion400m_e32=_pcfg(sha256='74799b8f05c38a6364ad999afb459b9fafa073e35fbbb063a6a79ab4d8f30ac9'),
    laion2b_s34b_b88k=_pcfg(sha256='fe66282f80d7243afb7a463db56cf0d3f1bcce0abf665b6af5134544be451700'),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(sha256='20d5f78b7ea7aad0f9727d7c0c1e27be06a16ecbb2130ada290e7a166b57ad23'),
    # DataComp-L models
    datacomp_l_s1b_b8k=_pcfg(sha256='3ef342da977ec12d8f2d25d2f3af9e0f98ecaaa0069aac255369bd09133dc8d8'),
    commonpool_l_clip_s1b_b8k=_pcfg(sha256='6a42ac82a9dda60ec082855cd746a90eb2dbc566fc5e47539cd6bcf4b4d5159b'),
    commonpool_l_laion_s1b_b8k=_pcfg(sha256='14f09c24bc8a7516d9147b34cb0bd7f058c43339e7c3b10cd10e55a1cc82b995'),
    commonpool_l_image_s1b_b8k=_pcfg(sha256='12679e00d2be224de4eb971534b0e64943b151b8570387bd390c51b6dae3a45b'),
    commonpool_l_text_s1b_b8k=_pcfg(sha256='bb91fe106f4eed27dc672ad12ba59b27f43cff0651a27bceeffd7230a4307cff'),
    commonpool_l_basic_s1b_b8k=_pcfg(sha256='7f6df7729ebacd07d8a27bd69b5c99755731328ee81f5f53d5475fb120c67396'),
    commonpool_l_s1b_b8k=_pcfg(sha256='73f2da6dc01575c3ddb17ed1b459b1ef07dc9fe5dc54e68d64bcdd1c8407559a'),
    # DFN
    dfn2b=_pcfg(sha256='3cbd3380bb36041dac08f1f4a8c38fbed03f934163872e76db43ca164aed0c0d')
)

_VITB16_quickgelu = dict(
    # OpenAI models were trained with QuickGELU
    openai=_pcfg(sha256='a2f7a56319a2be0e8f226480ccf2d29ceb20e5e414bbd158a883664db9276cac'),
    metaclip_400m=_pcfg(sha256='2c186d9d1e2e69c7caf6bf3e2c076df9032ff53439a5b8df5411f829863b9963'),
    metaclip_fullcc=_pcfg(sha256='4575651e9fa5259fe12e34b747795fb7122e1fa3051eb51e40e25ca29aadba16'),
)

_VITB16_PLUS_240 = dict(
    laion400m_e31=_pcfg(sha256='6817324ef268d8cd790cd778b4f1f12458323cd7bba0edad4f9510582d0367f3'),
    laion400m_e32=_pcfg(sha256='c2f976d855c403cf4bce9d8e0dde9cff965d97e64864a061ae2b3bae6875594e'),
)

_VITL14 = dict(
    # openai=_pcfg(),  # OpenAI models were trained with QuickGELU
    laion400m_e31=_pcfg(sha256='42f8c8fc92bd60272eb6d3784df5fdfed8b3ba73f7d9d5c7be635a3465bd7150'),
    laion400m_e32=_pcfg(sha256='bc883c6f3bd007e6c0ee4c16bea68220b4f0d14deb3ba3c246ad8d954551e467'),
    laion2b_s32b_b82k=_pcfg(
        mean=INCEPTION_MEAN, std=INCEPTION_STD,
        sha256='6e882e18abf7bd6111aa3651dcbc265cb99b784ce40bb7b7e73ae0ed39586614'),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(sha256='a18b7ae3a2ef324fb5115b0f742ae15f49e1bd8318eb54371a2c45a1b681a1c0'),
    commonpool_xl_clip_s13b_b90k=_pcfg(sha256='1e1987048d3d2e008898f81c8ff54f04b9c12cf23dc9ce9006897d31ee53f157'),
    commonpool_xl_laion_s13b_b90k=_pcfg(sha256='1193dbce1dfcaef9f9005e2d931f22c9d677af02c18c89f973ab38b5aa82f3a4'),
    commonpool_xl_s13b_b90k=_pcfg(sha256='b7515ce1a933ad6d8c35ff0a8a7590f76e7a02a18bb71697da101ba1f36d4b39'),
)

_VITL14_quickgelu = dict(
    # OpenAI models were trained with QuickGELU
    openai=_pcfg(sha256='cd11e3ebc37d88a89e9fb788fe32472791de23813666b1531ab971d3aae327c5'),
    metaclip_400m=_pcfg(sha256='d992eb4050ed92f50fc92889e8a3417d1a35d338676930931544617c81244ec5'),
    metaclip_fullcc=_pcfg(sha256='383266d4e77954cd8602a92fb095103ed4d0118b51bb17d739d12a336cc7c149'),
    dfn2b=_pcfg(sha256='3506d4e53b02c5c8425a83e9def3c3bc85c33ad2b64f0c3670ed08842c26acd3'),
)

_VITL14_336_quickgelu = dict(
    # OpenAI models were trained with QuickGELU
    openai=_pcfg(sha256='ad628eb668585f44e59fc151da40f2fa47c4d4e32336259f8fc1522ee40088d9'),
)

_VITH14 = dict(
    laion2b_s32b_b79k=_pcfg(),
)

_VITH14_quickgelu = dict(
    metaclip_fullcc=_pcfg(),
    dfn5b=_pcfg(resize_mode='squash'),
)

_VITH14_378_quickgelu = dict(
    dfn5b=_pcfg(resize_mode='squash'),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(),
    laion2b_s34b_b88k=_pcfg(),
)

_VITbigG14 = dict(
    laion2b_s39b_b160k=_pcfg(),
)

_coca_VITB32 = dict(
    laion2b_s13b_b90k=_pcfg(sha256='ff9b00049daad95ffc724ae8b57f5760ea99e6f41b7e4ad8b69198198dd6f767'),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(sha256='e8d09d9feb7b6945ad241655639729bf72f54efeda1f153b05b71b508be435fa')
)

_coca_VITL14 = dict(
    laion2b_s13b_b90k=_pcfg(sha256='1a42f67fb75668e55c2a507022ea490de7ee946786afbe37cb6b992385440476'),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(sha256='fc85f824dd2a70be272652b7fa83327e96b91675a8431cfe843d13ab69043411')
)

_PRETRAINED = {
    'ViT-B-32': _VITB32,
    'ViT-B-32-256': _VITB32_256,
    'ViT-B-32-quickgelu': _VITB32_quickgelu,
    'ViT-B-16': _VITB16,
    'ViT-B-16-quickgelu': _VITB16_quickgelu,
    'ViT-B-16-plus-240': _VITB16_PLUS_240,
    'ViT-L-14': _VITL14,
    'ViT-L-14-quickgelu': _VITL14_quickgelu,
    'ViT-L-14-336-quickgelu': _VITL14_336_quickgelu,  # OpenAI models were trained with QuickGELU
    'ViT-H-14': _VITH14,
    'ViT-H-14-quickgelu': _VITH14_quickgelu,
    'ViT-H-14-378-quickgelu': _VITH14_378_quickgelu,
    'ViT-g-14': _VITg14,
    'ViT-bigG-14': _VITbigG14,

    'coca_ViT-B-32': _coca_VITB32,
    'coca_ViT-L-14': _coca_VITL14,

    'EVA01-g-14': dict(
        # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_psz14_s11B.pt
        laion400m_s11b_b41k=_pcfg(),
    ),
    'EVA01-g-14-plus': dict(
        # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_plus_psz14_s11B.pt
        merged2b_s11b_b114k=_pcfg(),
    ),
    'EVA02-B-16': dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_B_psz16_s8B.pt
        merged2b_s8b_b131k=_pcfg(sha256='d4436d70259eb6c0628567179a0adb86955d82a3378003e6baf43655d884ba21'),
    ),
    'EVA02-L-14': dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_s4B.pt
        merged2b_s4b_b131k=_pcfg(sha256='b2deef65d02f4617bac637bf2a313663be29c5e651f03efb7f5d84a804d3d54c'),
    ),
    'EVA02-L-14-336': dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt
        merged2b_s6b_b61k=_pcfg(sha256='3c3826c2ac3e9b3fbcba9e7ad4596646aba4ab5c0d0fe9556d482d1a27c289e6'),
    ),
    'EVA02-E-14': dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_s4B.pt
        laion2b_s4b_b115k=_pcfg(),
    ),
    'EVA02-E-14-plus': dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt
        laion2b_s9b_b144k=_pcfg(),
    ),

    'ViT-B-16-SigLIP': dict(
        webli=_slpcfg(sha256='cb71518557f6e047e24d92aeb72e78b519b151dadaed5123d28cb71940cf4c66'),
    ),
    'ViT-B-16-SigLIP-256': dict(
        webli=_slpcfg(sha256='850a9b318b893070d4a0b35ddb237b50281fc155d84b234e463c1d3714e88505'),
    ),
    'ViT-B-16-SigLIP-i18n-256': dict(
        webli=_slpcfg(sha256='9bd33305441b7603543d13142752c0407cecda677a5f94bd025f0754a20eb50b'),
    ),
    'ViT-B-16-SigLIP-384': dict(
        webli=_slpcfg(sha256='9637f1ed2ccfbbccf54bc0638c366c7e42639c0374b5d90930dfd7c3f3a159d7'),
    ),
    'ViT-B-16-SigLIP-512': dict(
        webli=_slpcfg(sha256='ab13bcfc5c95080056a83adc37ecbdbf9877a5e3c4e4f20c203c7e7e10bd5e75'),
    ),
    'ViT-L-16-SigLIP-256': dict(
        webli=_slpcfg(),
    ),
    'ViT-L-16-SigLIP-378': dict(
        webli=_slpcfg(),
    ),
    'ViT-SO400M-14-SigLIP': dict(
        webli=_slpcfg(),
    ),
    'ViT-SO400M-14-SigLIP-378': dict(
        webli=_slpcfg(),
    ),

    'ViT-L-14-CLIPA': dict(
        datacomp1b=_apcfg(sha256='c6b750409035b20d8d43a260a590a643cd2bdce7e00a19074266228e799b36da'),
    ),
    'ViT-L-14-CLIPA-336': dict(
        datacomp1b=_apcfg(sha256='10c59fa86d38786927edf6b8a9c40ae65763ec76026e13f84c383e47f1ba722a'),
    ),
    'ViT-H-14-CLIPA': dict(
        datacomp1b=_apcfg(),
    ),
    'ViT-H-14-CLIPA-336-quickgelu': dict(
        laion2b=_apcfg(),
        datacomp1b=_apcfg(),
    ),
    'ViT-bigG-14-CLIPA': dict(
        datacomp1b=_apcfg(),
    ),
    'ViT-bigG-14-CLIPA-336': dict(
        datacomp1b=_apcfg(),
    ),
}


def _clean_tag(tag):
    return tag.lower().replace('-', '_')


def list_pretrained(as_str=False):
    return [':'.join([k, t]) if as_str else (k, t) for k in _PRETRAINED.keys() for t in _PRETRAINED[k].keys()]


def list_pretrained_models_by_tag(tag):
    models = []
    tag = _clean_tag(tag)
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)

    return models


def list_pretrained_tags_by_model(model):
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def get_pretrained_cfg(model, tag):
    if model not in _PRETRAINED:
        return {}
    if tag is None:
        return {}

    model_pretrained = _PRETRAINED[model]

    return model_pretrained.get(_clean_tag(tag), {})


def get_pretrained_url(model, tag):
    tag = _clean_tag(tag)
    cfg = get_pretrained_cfg(model, tag)

    version = cfg.get('version', None)
    if version is None:
        return None

    sha256 = cfg.get('sha256', None)
    if sha256 is None:
        return None

    return f'https://github.com/shkarupa-alex/tfclip/releases/download/{version}/{model}__{tag}__{sha256}.h5'
