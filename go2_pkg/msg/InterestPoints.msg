# go2_pkg/msg/InterestPoints.msg
# One flat array per field for efficiency
std_msgs/Header header
float32[] xy          # [x0,y0,  x1,y1, ...] length = 2*N
float32[] descriptors # flattened row-major (N*256)
float32[] scores      # length = N (SuperPoint confidences)
