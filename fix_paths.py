#!/usr/bin/env python3
"""
Script to fix file paths in list files for Joint-VAD dataset
"""
import os
import glob

def fix_list_files():
    # 수정할 리스트 파일들 찾기
    list_files = [
        'data/weakly_ShanghaiTech/list/shanghai-i3d-train-10crop.list',
        'data/weakly_ShanghaiTech/list/shanghai-i3d-test-10crop.list',
        'data/weakly_UBnormal/list/ubnormal-i3d-train-10crop.list',
        'data/weakly_UBnormal/list/ubnormal-i3d-test-10crop.list',
        'data/weakly_ShanghaiTech/list/shanghai-c3d-train-10crop.list',
        'data/weakly_ShanghaiTech/list/shanghai-c3d-test-10crop.list',
        'data/weakly_UBnormal/list/ubnormal-c3d-train-10crop.list',
        'data/weakly_UBnormal/list/ubnormal-c3d-test-10crop.list'
    ]
    
    # 새로운 기본 경로 설정 (현재 작업 디렉토리 기준)
    current_dir = os.getcwd()
    
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
                
                # 새 경로 생성
                if 'shanghai' in list_file.lower():
                    if 'train' in list_file:
                        if 'i3d' in list_file:
                            new_path = os.path.join(current_dir, 'data/weakly_ShanghaiTech/features/SH_Train_ten_crop_i3d', filename)
                        elif 'c3d' in list_file:
                            new_path = os.path.join(current_dir, 'data/weakly_ShanghaiTech/features/SH_Train_ten_crop_c3d', filename)
                    elif 'test' in list_file:
                        if 'i3d' in list_file:
                            new_path = os.path.join(current_dir, 'data/weakly_ShanghaiTech/features/SH_Test_ten_crop_i3d', filename)
                        elif 'c3d' in list_file:
                            new_path = os.path.join(current_dir, 'data/weakly_ShanghaiTech/features/SH_Test_ten_crop_c3d', filename)
                elif 'ubnormal' in list_file.lower():
                    if 'train' in list_file:
                        if 'i3d' in list_file:
                            new_path = os.path.join(current_dir, 'data/weakly_UBnormal/features/UB_Train_ten_crop_i3d', filename)
                        elif 'c3d' in list_file:
                            new_path = os.path.join(current_dir, 'data/weakly_UBnormal/features/UB_Train_ten_crop_c3d', filename)
                    elif 'test' in list_file:
                        if 'i3d' in list_file:
                            new_path = os.path.join(current_dir, 'data/weakly_UBnormal/features/UB_Test_ten_crop_i3d', filename)
                        elif 'c3d' in list_file:
                            new_path = os.path.join(current_dir, 'data/weakly_UBnormal/features/UB_Test_ten_crop_c3d', filename)
                
                new_lines.append(new_path + '\n')
            else:
                new_lines.append(line + '\n')
        
        # 파일 쓰기
        with open(list_file, 'w') as f:
            f.writelines(new_lines)
        
        print(f"✅ 완료: {len(new_lines)}개 경로 수정")

def check_paths():
    """수정된 경로들이 실제로 존재하는지 확인"""
    print("\n🔍 경로 존재 여부 확인...")
    
    list_files = [
        'data/weakly_ShanghaiTech/list/shanghai-i3d-train-10crop.list',
        'data/weakly_ShanghaiTech/list/shanghai-i3d-test-10crop.list'
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
            print("💡 I3D features를 다운로드하고 올바른 위치에 배치하세요.")

if __name__ == "__main__":
    print("🚀 Joint-VAD 경로 수정 스크립트")
    print("=" * 50)
    
    # 1. 경로 수정
    fix_list_files()
    
    # 2. 확인
    check_paths()
    
    print("\n" + "=" * 50)
    print("✨ 완료! 백업 파일들(.backup)이 생성되었습니다.")
    print("💡 문제가 있으면 백업 파일로 복원할 수 있습니다.")