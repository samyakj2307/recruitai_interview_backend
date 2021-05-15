from rest_framework.views import APIView
from . import interview_process_runner
from django.http import JsonResponse
from rest_framework import status

class VideoProcessing(APIView):

    def get(self, request):
        if request.method == 'GET':
            userid = request.GET.get('userid')
            companyid = request.GET.get('companyid')
            interview_process_runner.process_video.delay(userid,companyid)

            response = {
                'id': userid,
                'status': "Video Processing Started",
            }
            return JsonResponse(response, status=status.HTTP_200_OK)