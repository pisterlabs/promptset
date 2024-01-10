from rest_framework import routers
from core.user.viewsets import UserViewSet
from core.auth.viewsets import RegisterViewSet, LoginViewSet, RefreshViewSet
from core.wallet import viewsets
router = routers.SimpleRouter()
from core.wallet.viewsets import TopUpBtcViewset
from email_msg_generator.viewsets import OpenAiUserViewSet
from core.user.viewsets import CurrentUserTokenViewset
from email_msg_generator.viewsets import OpenAiUserViewSet

# ##################################################################### #
# ################### AUTH                       ###################### #
# ##################################################################### #

router.register(r'auth/register', RegisterViewSet, basename='auth-register')
router.register(r'auth/login', LoginViewSet, basename='auth-login')
router.register(r'auth/refresh', RefreshViewSet, basename='auth-refresh')

# router.register()

# ##################################################################### #
# ################### USER                       ###################### #
# ##################################################################### #
from core.wallet.viewsets import BtcAmountView
router.register(r'user', UserViewSet, basename='user')
router.register(r'usdtopup', TopUpBtcViewset, basename='usdtopup')
# from core.user.viewsets import TokenStoredVieset
# router.register(r'openapikeyview',OpenAIViewset,basename='openapikeyview')
router.register(r'checktoken',CurrentUserTokenViewset,basename='checktoken')
# router.register(r'wallet_top_up',viewsets.Wallet,basename='wallet')
router.register(r'openaiusers', OpenAiUserViewSet, basename='openaiusers')
# router.register(r'tokenstored',TokenStoredVieset,basename='basename')
# router.register('emailmessagesending',EmailMessageViewsets,basename='emailmessagesending')

urlpatterns = [
    *router.urls,
]