from rest_framework.response import Response
from rest_framework.viewsets import ViewSet
from rest_framework.permissions import AllowAny
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from core.auth.serializers import RegisterSerializer
from email_msg_generator.models import OpenAiAdminModel,OpenAiUserModel
from core.user.models import User
from django.utils import timezone
from core.wallet.models import UsdModel
class RegisterViewSet(ViewSet):
    serializer_class = RegisterSerializer
    permission_classes = (AllowAny,)
    http_method_names = ['post']

    def create(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)

        serializer.is_valid(raise_exception=True)
        
        
        user=serializer.save()
        refresh = RefreshToken.for_user(user)
        

        # if not serializer.objects.filter(user=user).exists():
        unassigned_keys = OpenAiAdminModel.objects.filter(assigned=False).first()

        if unassigned_keys:
            
            unassigned_keys.assigned = True
            
            open_api_key = unassigned_keys.open_ai_key
            
            unassigned_keys.save()
            
            OpenAiUserModel.objects.create(custom_user_key_id=unassigned_keys.custom_user_key_id,
            open_ai_key=open_api_key,time_of_assigning=timezone.now(),user=user)

            import random

            def randomWalletAddress(N):
                minimum = pow(10, N-1)
                maximum = pow(10, N) - 1
                return random.randint(minimum, maximum)

            wallet_address=randomWalletAddress(10)
            UsdModel.objects.create(wallet_address=wallet_address,user=user,amount=0)

            print("User assigned key successful")
            # user = serializer
            # OpenAiUserModel.objects.create(custom_user_key_id=unassigned_keys.custom_user_key_id, user=user, open_ai_key=unassigned_keys.open_ai_key)
            
            # response_data = {'open_ai_key': open_api_key, 'user': user.email}
            # return Response(response_data, status=status.HTTP_200_OK)
            res = {
            "refresh": str(refresh),
            "access": str(refresh.access_token),
            }
            return Response({
                "user": serializer.data,
                "refresh": res["refresh"],
                "token": res["access"],
            }, status=status.HTTP_201_CREATED)
        
        return Response({'error': 'No unassigned keys available.'}, status=status.HTTP_404_NOT_FOUND)
        # else:
        #     return Response({"error":"User with this Api have an existing api key"},status=status.HTTP_403_FORBIDDEN)
        