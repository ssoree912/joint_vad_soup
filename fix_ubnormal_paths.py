#!/usr/bin/env python3
"""
Script to fix file paths in UBnormal list files for Joint-VAD dataset
"""
import os

def fix_ubnormal_paths():
    """UBnormal 데이터셋의 경로 수정"""
    
    # 수정할 UBnormal 리스트 파일들
    list_files = [
        'data/weakly_UBnormal/list/ubnormal-i3d-train-10crop.list',
        'data/weakly_UBnormal/list/ubnormal-i3d-test-10crop.list',
        'data/weakly_UBnormal/list/ubnormal-i3d-validation-10crop.list'
    ]
    
    current_dir = os.getcwd()
    
    print("🚀 UBnormal 경로 수정 스크립트")
    print("=" * 50)
    
    for list_file in list_files:
        if not os.path.exists(list_file):
            print(f"⚠️  파일이 없습니다: {list_file}")
            continue
            
        print(f"🔧 수정 중: {list_file}")
        
        # 백업 파일 생성
        backup_file = list_file + '.backup'
        if not os.path.exists(backup_file):
            os.system(f"cp '{list_file}' '{backup_file}'")
            print(f"📋 백업 생성: {backup_file}")
        
        # 파일 읽기
        with open(list_file, 'r') as f:
            lines = f.readlines()
        
        # 경로 수정
        new_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # 파일명만 추출
                filename = os.path.basename(line)
                
                # 새 경로 생성 (파일명에 따라 분류)
                if 'train' in list_file:
                    new_path = os.path.join(current_dir, 'data/weakly_UBnormal/features/UB_Train_ten_crop_i3d', filename)
                elif 'test' in list_file:
                    new_path = os.path.join(current_dir, 'data/weakly_UBnormal/features/UB_Test_ten_crop_i3d', filename)
                elif 'validation' in list_file:
                    new_path = os.path.join(current_dir, 'data/weakly_UBnormal/features/UB_Validation_ten_crop_i3d', filename)
                
                new_lines.append(new_path + '\n')
            else:
                new_lines.append(line + '\n')
        
        # 파일 쓰기
        with open(list_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"✅ 완료: {len([l for l in new_lines if l.strip() and not l.startswith('#')])}개 경로 수정")

def check_ubnormal_structure():
    """UBnormal 폴더 구조 확인"""
    print("\n📁 UBnormal 폴더 구조 확인...")
    
    base_path = "data/weakly_UBnormal/features"
    
    if os.path.exists(base_path):
        print(f"✅ 기본 경로 존재: {base_path}")
        subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        for subdir in subdirs:
            full_path = os.path.join(base_path, subdir)
            file_count = len([f for f in os.listdir(full_path) if f.endswith('.npy')])
            print(f"📂 {subdir}: {file_count}개 파일")
    else:
        print(f"❌ 기본 경로 없음: {base_path}")
        print("💡 UBnormal I3D features를 다운로드하고 올바른 위치에 배치하세요.")

def check_ubnormal_paths():
    """수정된 UBnormal 경로들이 실제로 존재하는지 확인"""
    print("\n🔍 UBnormal 경로 존재 여부 확인...")
    
    list_files = [
        'data/weakly_UBnormal/list/ubnormal-i3d-train-10crop.list',
        'data/weakly_UBnormal/list/ubnormal-i3d-test-10crop.list'
    ]
    
    for list_file in list_files:
        if not os.path.exists(list_file):
            continue
            
        print(f"\n📁 {list_file}")
        
        with open(list_file, 'r') as f:
            lines = f.readlines()
        
        missing_count = 0
        total_count = 0
        
        for line in lines[:5]:  # 처음 5개만 체크
            path = line.strip()
            if path and not path.startswith('#'):
                total_count += 1
                if not os.path.exists(path):
                    print(f"❌ 없음: {path}")
                    missing_count += 1
                else:
                    print(f"✅ 존재: {path}")
        
        if missing_count > 0:
            print(f"⚠️  {missing_count}/{total_count} 파일이 없습니다.")
        else:
            print(f"✅ 모든 파일 존재!")

def show_expected_structure():
    """예상되는 UBnormal 폴더 구조 표시"""
    print("\n📋 예상되는 UBnormal 폴더 구조:")
    print("data/weakly_UBnormal/features/")
    print("├── UB_Train_ten_crop_i3d/")
    print("├── UB_Test_ten_crop_i3d/")
    print("└── UB_Validation_ten_crop_i3d/")
    print("\n💡 UBnormal I3D features 다운로드 링크:")
    print("https://drive.google.com/file/d/1dHWrvO5ZDtmqvgqOpttRazI5HyxUAnQp/view?usp=sharing")

if __name__ == "__main__":
    # 1. 현재 구조 확인
    check_ubnormal_structure()
    
    # 2. 예상 구조 표시
    show_expected_structure()
    
    # 3. 경로 수정
    fix_ubnormal_paths()
    
    # 4. 확인
    check_ubnormal_paths()
    
    print("\n" + "=" * 50)
    print("✨ UBnormal 경로 수정 완료!")
    print("💡 UBnormal으로 훈련하려면:")
    print("python main.py --dataset UBnormal --ab_ratio 0.3 --seg_len 16")