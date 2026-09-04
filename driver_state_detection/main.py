import pprint

from driver_state_detection.qt_compat import configure_qt_before_cv2_import

configure_qt_before_cv2_import()

import cv2

from driver_state_detection.parser import get_args
from driver_state_detection.qt_compat import configure_qt_fonts_after_cv2_import
from driver_state_detection.utils import load_camera_parameters

configure_qt_fonts_after_cv2_import()


def main(argv=None):
    """Run webcam detection and own the lifetime of camera and GUI resources."""
    from ergosense.presentation.opencv_monitor import run_opencv_monitoring

    args = get_args(argv)
    if not args.model_path.is_file():
        raise SystemExit(
            f"Face Landmarker model not found at {args.model_path}. Run "
            "`driver-state-detection-download-model` first."
        )

    if not cv2.useOptimized():
        cv2.setUseOptimized(True)

    try:
        camera_matrix, dist_coeffs, camera_image_size = (
            load_camera_parameters(args.camera_params)
            if args.camera_params
            else (None, None, None)
        )
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"Invalid camera parameters: {error}") from error

    if args.verbose:
        print("Arguments and parameters:")
        pprint.pp(vars(args), indent=4)
        print("Camera matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("Distortion coefficients:")
        pprint.pp(dist_coeffs, indent=4)

    run_opencv_monitoring(
        args,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        camera_image_size=camera_image_size,
    )


if __name__ == "__main__":
    main()
